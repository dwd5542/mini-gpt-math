# 새 프로젝트 시작용 환경 설정 스크립트
# 사용법: 새 프로젝트 폴더에서 .\setup_env.ps1 실행

Write-Host "Python 3.12로 venv 생성 중..."
py -3.12 -m venv venv

Write-Host "venv 활성화 중..."
& venv\Scripts\Activate.ps1

Write-Host "pip 최신화 및 필수 라이브러리 설치 중..."
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install transformers datasets scikit-learn matplotlib numpy pandas

Write-Host "GPU 확인 중..."
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

Write-Host "`.gitignore 생성 중..."
Set-Content -Path .gitignore -Encoding utf8 -Value "venv/`n__pycache__/`n*.npy`n.cache/`n*.pyc"

Write-Host "git 초기화 중..."
git init

Write-Host "완료!"