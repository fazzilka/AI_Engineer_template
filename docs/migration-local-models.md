# Migration to local models

This revision intentionally removes the previous OpenAI-compatible runtime and the `openai` SDK.

Before:

```dotenv
LLM__PROVIDER=openai_compatible
LLM__API_KEY=...
LLM__BASE_URL=...
```

After:

```dotenv
MODEL__BACKEND=huggingface
MODEL__SOURCE=filesystem
MODEL__PATH=./models/generator
MODEL__LOCAL_FILES_ONLY=true
MODEL__TRUST_REMOTE_CODE=false
```

Migration steps:

1. Remove old provider credentials from deployment secret stores after verifying no other service uses
   them.
2. Choose and license generator/embedding models; pin immutable revisions.
3. Run the explicit download command outside normal install/CI.
4. Create a new Qdrant collection or reindex with the new embedding fingerprint.
5. Run `make check`, `make eval`, and the opt-in local model smoke.
6. Re-size CPU/RAM/GPU resources: model memory now belongs to the application process.

The `/api/v1/chat` messages contract remains stateless and substantially compatible. Provider selection,
credentials, base URL, and remote retry settings no longer exist.
