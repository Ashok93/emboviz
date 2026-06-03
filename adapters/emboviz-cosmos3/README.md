# emboviz-cosmos3

NVIDIA **Cosmos3-Nano** world-model adapter for emboviz — action-conditioned
**forward dynamics**: given a conditioning frame and a sequence of robot
actions, generate the future video those actions produce.

This is a world model (`emboviz_wire.world_model_protocol.WorldModel`), not a
policy. It is the substrate for emboviz's trust-calibration and rollout
diagnostics: feed a recorded episode's real actions, compare the predicted
rollout against what actually happened.

## How it runs

Action conditioning is served only by NVIDIA's **vLLM-Omni** server, which
holds the BF16 model on its own GPU. This adapter is a **thin HTTP client** —
no torch, no GPU. Two pieces run side by side:

```
GPU box / container
├── vllm/vllm-omni:cosmos3   →  serves /v1/videos/sync   (the 33 GB model)
└── emboviz-cosmos3 worker   →  POSTs frame + actions, decodes the MP4 rollout
```

Start the server (BF16 only — FP8/NVFP4 are not supported for the action path):

```bash
vllm serve nvidia/Cosmos3-Nano --omni --host 0.0.0.0 --port 8000 --init-timeout 1800
```

Then point the worker at it, specifying the embodiment (`domain_name`) and its
action dimensionality (`action_dim`) — both required, never guessed:

```bash
emboviz-cosmos3 serve --sock /tmp/emboviz/cosmos3.sock \
  --kwargs '{"server_url": "http://localhost:8000",
             "domain_name": "agibotworld", "action_dim": 29}'
```

## Action normalization

Cosmos conditions on actions in its own per-domain normalized space. Supply an
`action_normalizer` callable to map native actions into that space; with the
default (`None`), actions are passed through unchanged — i.e. the caller must
provide actions already in the domain's normalized convention. No normalization
is inferred or guessed.

## License

Apache-2.0. The Cosmos3-Nano weights are governed by NVIDIA's own license; this
adapter downloads and redistributes nothing.
