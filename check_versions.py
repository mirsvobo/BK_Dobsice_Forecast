import importlib.metadata
import sys

def check_package_versions():
    """
    Vypíše všechny nainstalované balíčky a jejich verze.
    """
    print(f"--- Kontroluji balíčky pro Python {sys.version.split()[0]} ---")
    print("="*60)

    try:
        # Získáme seznam všech nainstalovaných balíčků
        installed_packages = importlib.metadata.distributions()

        # Seřadíme je podle jména
        package_list = sorted(
            [(dist.metadata['name'], dist.version) for dist in installed_packages]
        )

        # Vytiskneme je
        if not package_list:
            print("CHYBA: Nebyly nalezeny žádné balíčky.")
            print("Je virtuální prostředí (.venv) správně aktivováno?")
            return

        print(f"Nalezeno {len(package_list)} balíčků:\n")
        for name, version in package_list:
            print(f"{name:<30} == {version}")

    except Exception as e:
        print(f"CHYBA: Nepodařilo se získat seznam balíčků.")
        print(f"Důvod: {e}")

    print("="*60)
    print("--- Kontrola dokončena ---")

if __name__ == "__main__":
    check_package_versions()
    check_package_versions()