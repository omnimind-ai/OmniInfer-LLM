from executorch.exir._serialize._program import deserialize_pte_binary


pte_path = "/home/syf/executorch/qwen_qnn_1202/hybrid_llama_qnn.pte"
with open(pte_path, "rb") as f:
    program_data = f.read()
program = deserialize_pte_binary(program_data)
for method in program.execution_plan:
    if method.name == "get_vocab_size":
        pass