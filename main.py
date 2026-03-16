import subprocess
import sys
import os

def main():
    # Caminho para o arquivo streamlit_page.py
    streamlit_file = os.path.join(os.path.dirname(__file__), 'resources', 'streamlit_page.py')

    # Executar streamlit run
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', streamlit_file])

if __name__ == "__main__":
    main()