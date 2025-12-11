import torch
import sys
import subprocess
import os

def get_system_info_markdown():
    lines = []

    # 1. PYTHON
    lines.append(f"**Python:** `{sys.version.split()[0]}`")

    # 2. PYTORCH INTERNALS
    torch_version = torch.__version__
    lines.append(f"**PyTorch Verze:** `{torch_version}`")

    cuda_available = torch.cuda.is_available()

    # Zjištění verze CUDA, se kterou byl PyTorch sestaven
    torch_cuda_version = torch.version.cuda
    lines.append(f"**PyTorch Built with CUDA:** `{torch_cuda_version}`")

    if cuda_available:
        lines.append("\n✅ **GPU Akcelerace: AKTIVNÍ**")

        # GPU Info
        try:
            device_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / 1e9

            lines.append(f"🖥️ GPU: `{device_name}`")
            lines.append(f"💾 VRAM: `{vram_gb:.2f} GB`")

            # Tensor Cores Check
            major = props.major
            if major >= 7:
                lines.append("🚀 Tensor Cores: `ANO` (Architektura podporována)")
            else:
                lines.append("⚠️ Tensor Cores: `NE` (Starší architektura)")
        except Exception as e:
            lines.append(f"⚠️ Chyba čtení GPU: {e}")

        lines.append("\n--- **CUDA & DRIVER CHECK** ---")

        # 3. NVIDIA-SMI (Driver Version & Max Supported CUDA)
        try:
            # Získáme verzi driveru
            driver = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                encoding='utf-8'
            ).strip()
            lines.append(f"🔧 **NVIDIA Driver:** `{driver}`")

            # Získáme CUDA verzi z nvidia-smi (to je verze, kterou driver podporuje)
            # Nvidia-smi header obsahuje "CUDA Version: XX.X"
            smi_out = subprocess.check_output(['nvidia-smi'], encoding='utf-8')
            for line in smi_out.split('\n'):
                if "CUDA Version" in line:
                    sys_cuda = line.split("CUDA Version:")[1].split()[0].strip()
                    lines.append(f"🔌 **Driver Supported CUDA:** `{sys_cuda}`")
                    break
        except FileNotFoundError:
            lines.append("❌ `nvidia-smi` nenalezeno (jsou ovladače v PATH?)")
        except Exception as e:
            lines.append(f"⚠️ Chyba nvidia-smi: {e}")

        # 4. NVCC (System CUDA Toolkit - volitelné)
        try:
            nvcc_out = subprocess.check_output(['nvcc', '--version'], encoding='utf-8')
            # Hledáme řádek s verzí, např. "release 12.4,"
            import re
            match = re.search(r"release (\d+\.\d+)", nvcc_out)
            if match:
                lines.append(f"🛠️ **System NVCC Toolkit:** `{match.group(1)}`")
            else:
                lines.append(f"🛠️ System NVCC: Detekováno, ale verze nepřečtena")
        except FileNotFoundError:
            lines.append("ℹ️ System NVCC: `Nenalezeno` (Nevadí, PyTorch má vlastní runtime)")

        # 5. KOMPATIBILITA CHECK
        lines.append("\n**Verdikt:**")

        # Logika: Driver CUDA musí být >= PyTorch CUDA
        try:
            if 'sys_cuda' in locals() and torch_cuda_version:
                sys_ver = float(sys_cuda.split('.')[0] + "." + sys_cuda.split('.')[1])
                torch_ver = float(torch_cuda_version.split('.')[0] + "." + torch_cuda_version.split('.')[1])

                if sys_ver >= torch_ver:
                    lines.append("✅ **OK:** Verze ovladače podporuje verzi PyTorch.")
                else:
                    lines.append(f"⚠️ **POZOR:** Ovladač podporuje max CUDA {sys_ver}, ale PyTorch chce {torch_ver}.")
                    lines.append("   -> Může to fungovat (PyTorch si nese vlastní DLL), ale doporučuje se update driveru.")
        except:
            lines.append("ℹ️ Nelze automaticky ověřit kompatibilitu verzí (chybí data).")

    else:
        lines.append("❌ **GPU Akcelerace: NEAKTIVNÍ**")
        lines.append("⚠️ PyTorch nevidí GPU. Zkontroluj instalaci.")

    return "\n\n".join(lines)

if __name__ == "__main__":
    print(get_system_info_markdown())