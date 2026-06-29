import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from agent.model import DQN

def main():
    dim_estado = 36 
    dim_acoes = 3
    
    model = DQN(dim_estado=dim_estado, dim_acoes=dim_acoes)
    
    weights_path = os.path.join("data", "weights", "dqn_policy.pt")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    
    if isinstance(checkpoint, dict) and "policy_net" in checkpoint:
        model.load_state_dict(checkpoint["policy_net"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    dummy_input = torch.randn(1, dim_estado, dtype=torch.float32)
    
    onnx_path = os.path.join("data", "weights", "dqn_policy.onnx")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['estado'],
        output_names=['q_values'],
        dynamic_axes={'estado': {0: 'batch_size'}, 'q_values': {0: 'batch_size'}}
    )
    print(f"Modelo ONNX gerado com sucesso em: {onnx_path}")

if __name__ == "__main__":
    main()
