"""Private elephant-1 NF4 loader. No BF16 forward, no disk patch, no allocator.

Use only in a single-threaded loading process: the temporary classmethod wrapper
is process-global while active and always removed/restored before returning.
"""
import contextlib
import copy
import json
from pathlib import Path
import re
import shutil

SKIP = ['lm_head', 'in_proj_a', 'in_proj_b', 'conv1d']

@contextlib.contextmanager
def inject_quantization(cls, quantization_config):
    missing = object(); descriptor = cls.__dict__.get('from_pretrained', missing)
    original = cls.from_pretrained
    calls = []
    def wrapped(subclass, *args, **kwargs):
        if 'quantization_config' in kwargs:
            raise ValueError('Unexpected preexisting quantization config')
        calls.append(args)
        if len(calls) != 1: raise ValueError('Expected exactly one model load')
        kwargs['quantization_config'] = quantization_config
        return original(*args, **kwargs)
    cls.from_pretrained = classmethod(wrapped)
    try:
        yield calls
    finally:
        if descriptor is missing: delattr(cls, 'from_pretrained')
        else: setattr(cls, 'from_pretrained', descriptor)

def expected_shapes(config):
    c = config.get_text_config() if hasattr(config, 'get_text_config') else config
    expected = {'hidden_size':4096,'intermediate_size':12288,'num_hidden_layers':32,
                'num_attention_heads':16,'num_key_value_heads':4,'head_dim':256,
                'linear_key_head_dim':128,'linear_num_key_heads':16,
                'linear_num_value_heads':32,'linear_value_head_dim':128,'vocab_size':248320}
    for key, value in expected.items():
        if getattr(c,key,None) != value: raise ValueError('Unexpected elephant text dimension: '+key)
    types = list(c.layer_types)
    if types != ['full_attention' if i%4==3 else 'linear_attention' for i in range(32)]:
        raise ValueError('Unexpected elephant layer types')
    result = {}
    for i,t in enumerate(types):
        for name,shape in {'gate_proj':(4096,12288),'up_proj':(4096,12288),'down_proj':(12288,4096)}.items():
            result[f'{i}.mlp.{name}']=shape
        branches = ({'q_proj':(4096,8192),'k_proj':(4096,1024),'v_proj':(4096,1024),'o_proj':(4096,4096)}
                    if t=='full_attention' else {'in_proj_qkv':(4096,8192),'in_proj_z':(4096,4096),'out_proj':(4096,4096)})
        kind = 'self_attn' if t=='full_attention' else 'linear_attn'
        for name,shape in branches.items():result[f'{i}.{kind}.{name}']=shape
    return result

def validate_engine(engine, torch, linear4bit_class):
    model = engine.model; expected = expected_shapes(model.config); found = {}; protected = {}
    modules = dict(model.named_modules())
    for path,module in modules.items():
        if isinstance(module,linear4bit_class):
            match = re.fullmatch(r'(?:.+\.)?layers\.(\d+\.(?:mlp|self_attn|linear_attn)\.[^.]+)',path)
            if not match or match[1] not in expected or match[1] in found:
                raise ValueError('Unexpected/duplicate Linear4bit path: '+path)
            key=match[1]
            if (module.in_features,module.out_features)!=expected[key]:raise ValueError('Quantized dimensions mismatch: '+path)
            state=getattr(module.weight,'quant_state',None)
            if state is None or state.quant_type!='nf4' or getattr(state,'nested',False):
                raise ValueError('Expected initialized NF4 without double quantization: '+path)
            if module.compute_dtype!=torch.bfloat16:raise ValueError('Expected BF16 NF4 compute')
            found[key]=path
        if path.endswith(('.in_proj_a','.in_proj_b')):
            if module.weight.dtype!=torch.bfloat16:raise ValueError('Protected a/b projection is not BF16: '+path)
            if (module.in_features,module.out_features)!=(4096,32):raise ValueError('Protected a/b dimensions mismatch')
            protected[path]=str(module.weight.dtype)
    if set(found)!=set(expected) or len(found)!=200:raise ValueError(f'Expected exact 200 Linear4bit modules, got {len(found)}')
    if len(protected)!=48:raise ValueError('Expected all 48 protected a/b projections')
    if model.get_output_embeddings().weight.dtype!=torch.float32:raise ValueError('LM head must remain FP32')
    if model.get_input_embeddings().weight.dtype!=torch.bfloat16:raise ValueError('Embedding must remain BF16')
    if not engine.readout.config.noncausal or engine.readout.config.readout!='slot_lm':raise ValueError('Expected noncausal slot_lm')
    if engine.info.fork_mode!='sequential':raise ValueError('Expected sequential execution')
    return {'linear4bit_count':len(found),'linear4bit_paths':list(found.values()),'protected_ab_count':len(protected),
            'format':'nf4','double_quant':False,'compute_dtype':'bfloat16','head_dtype':'float32'}

def load(snapshot, *, device='cuda:0'):
    import torch
    import transformers
    from bitsandbytes.nn import Linear4bit
    from qev.engine import DecisionEngine
    snapshot=Path(snapshot).resolve()
    if not snapshot.is_dir():raise ValueError('Existing pinned local snapshot required')
    config=json.loads((snapshot/'config.json').read_text())
    if config.get('model_type') not in ('qwen3_5','qwen3_5_text'):raise ValueError('Unexpected source architecture')
    quant=transformers.BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=False,bnb_4bit_compute_dtype=torch.bfloat16,llm_int8_skip_modules=SKIP.copy())
    with inject_quantization(transformers.Qwen3_5ForCausalLM,quant) as calls:
        engine=DecisionEngine(str(snapshot),device=device,dtype='bfloat16',head_dtype='float32',
            fork_mode='sequential',max_branch_tokens=16384,max_request_tokens=16384)
        if len(calls)!=1:raise ValueError('Model load did not use the expected class')
    audit=validate_engine(engine,torch,Linear4bit)
    # Engine selects a text-only config. Preserve that real architecture while
    # carrying the serving mask metadata from the original multimodal wrapper.
    overrides=config.get('modeling_overrides')
    if not overrides or overrides.get('attention',{}).get('full_attention')!='block_bidirectional':
        raise ValueError('Missing trained block-bidirectional override')
    engine.model.config.modeling_overrides=copy.deepcopy(overrides)
    return engine,audit

def load_quantized_engine(snapshot, *, device='cuda:0'):
    engine,audit=load(snapshot,device=device)
    engine.quantization_audit=audit
    return engine

def save(engine, snapshot, destination, audit):
    source=Path(snapshot); dest=Path(destination)
    for required in ('readout.json','serve.patch','setup.sh','constraints.txt'):
        if not (source/required).is_file():raise ValueError('Missing serving metadata: '+required)
    readout=json.loads((source/'readout.json').read_text())
    if readout.get('readout','slot_lm')!='slot_lm':raise ValueError('Wrong source readout')
    readout['attention']='noncausal'
    dest.mkdir(exist_ok=False)  # Never overwrite an existing quantization artifact.
    engine.model.save_pretrained(str(dest),safe_serialization=True)
    engine.tokenizer.save_pretrained(str(dest))
    # Do NOT copy original config.json: it describes a multimodal wrapper, while
    # this saved model is the actual Qwen3_5ForCausalLM text model.
    (dest/'readout.json').write_text(json.dumps(readout,indent=2))
    for name in ('serve.patch','setup.sh','constraints.txt','calibration.json','chat_template.jinja'):
        if (source/name).is_file():shutil.copyfile(source/name,dest/name)
    saved=json.loads((dest/'config.json').read_text())
    if saved.get('model_type')!='qwen3_5_text':raise ValueError('Saved architecture is not the actual text model')
    if saved.get('modeling_overrides',{}).get('attention',{}).get('full_attention')!='block_bidirectional':
        raise ValueError('Saved config lost attention override')
    q=saved.get('quantization_config',{})
    if not q.get('load_in_4bit') or q.get('bnb_4bit_quant_type')!='nf4' or q.get('bnb_4bit_use_double_quant'):
        raise ValueError('Saved quantization configuration mismatch')
    (dest/'nf4_audit.json').write_text(json.dumps(audit,indent=2))
    return dest
