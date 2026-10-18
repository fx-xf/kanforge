# Transformers

Put KAN layers into a Hugging Face GPT-2 without changing its outputs, then fine-tune only the splines.

```bash
python examples/06_transformer_kanify.py
```

Set `KANFORGE_QUICK=1` for a few-second run (what the test suite does).

```python
--8<-- "examples/06_transformer_kanify.py"
```
