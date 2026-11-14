# Attribution and licenses

Quantized derivative of Base Labs / Baseten `baselabs/elephant-1`, revision `b3f8a93462fb1051a9de031e94a7ff380e254072`, whose model card declares MIT. Original model card is preserved in `upstream/README.baselabs.md`.

Base model: Qwen/Qwen3.5-9B (Apache-2.0); its license is retained in `upstream/LICENSE.Qwen`. The external OneJev runtime is Apache-2.0; license retained in `upstream/LICENSE.OneJev`. These upstream terms and notices continue to apply. This is an independent quantization, not an official Baseten release or endorsement.

Changes: language-model-only checkpoint, selected linear weights quantized to non-nested NF4, BF16 compute and embeddings, FP32 LM head; preserved decision attention/readout metadata. Added Agentware NF4 loader, provenance and usage documentation. No image encoder is included.
