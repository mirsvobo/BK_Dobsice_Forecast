import torch
import sys
import subprocess

def get_system_info_markdown():
    """
    Vrátí Markdown string s informacemi o systému pro Streamlit sidebar.
    """
    lines = []

    # Python
    lines.append(f"**Python:** `{sys.version.split()[0]}`")

    # PyTorch & CUDA
    lines.append(f"**PyTorch:** `{torch.__version__}`")
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        lines.append("✅ **GPU Akcelerace: AKTIVNÍ**")
        device_name = torch.cuda.get_device_name(0)
        lines.append(f"🖥️ `{device_name}`")

        # VRAM
        try:
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / 1e9
            lines.append(f"💾 VRAM: `{vram_gb:.1f} GB`")

            # Tensor Cores
            major = props.major
            if major >= 7:
                lines.append("🚀 Tensor Cores: `ANO`")
            else:
                lines.append("⚠️ Tensor Cores: `NE`")
        except:
            pass

        # Driver info
        try:
            smi = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], encoding='utf-8').strip()
            lines.append(f"🔧 Driver: `{smi}`")
        except:
            lines.append("🔧 Driver: `Neznámý`")

    else:
        lines.append("❌ **GPU Akcelerace: NEAKTIVNÍ**")
        lines.append("⚠️ *Jedeš na CPU. Bude to pomalé.*")
        lines.append("[Návod na instalaci CUDA](https://pytorch.org/get-started/locally/)")

    return "\n\n".join(lines)

if __name__ == "__main__":
    print(get_system_info_markdown())