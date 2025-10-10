import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import ChebConv
from tsl.data import SpatioTemporalDataset

def test_cuda():
    """Test basic CUDA availability"""
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    return torch.cuda.is_available()

def test_tensor_operations():
    """Test basic tensor operations on GPU"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping tensor tests")
        return False
    
    print("\n🧪 Testing tensor operations on GPU...")
    try:
        # Test basic tensor creation and operations
        x = torch.randn(1000, 1000, requires_grad=True).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✅ Basic tensor operations work")
        
        # Test gradient computation
        loss = z.sum()
        loss.backward()
        print("✅ Gradient computation works")
        
        return True
    except Exception as e:
        print(f"❌ Tensor operations failed: {e}")
        return False

def test_pytorch_geometric():
    """Test PyTorch Geometric on GPU"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping PyG tests")
        return False
    
    print("\n🧪 Testing PyTorch Geometric on GPU...")
    try:
        # Create a simple graph
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long).cuda()
        x = torch.randn(3, 4).cuda()  # 3 nodes, 4 features each
        
        # Test ChebConv (what you use in your model)
        conv = ChebConv(4, 8, K=2).cuda()
        out = conv(x, edge_index)
        print("✅ ChebConv works on GPU")
        
        # Test gradient computation
        out.sum().backward()
        print("✅ PyG gradient computation works")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch Geometric failed: {e}")
        return False

def main():
    """Run all GPU tests"""
    print("🚀 GPU Setup Verification")
    print("=" * 50)
    
    # Run all tests
    cuda_available = test_cuda()
    tensor_works = test_tensor_operations() if cuda_available else False
    pyg_works = test_pytorch_geometric() if cuda_available else False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY:")
    print(f"CUDA Available: {'✅' if cuda_available else '❌'}")
    print(f"Tensor Operations: {'✅' if tensor_works else '❌'}")
    print(f"PyTorch Geometric: {'✅' if pyg_works else '❌'}")
    
    if all([cuda_available, tensor_works, pyg_works]):
        print("\n🎉 All tests passed! You're ready for GPU training!")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
    
    return all([cuda_available, tensor_works, pyg_works])

if __name__ == "__main__":
    main()