# We are going to install PyTorch with CUDA 11.8 (or 12.1) support and ultralytics
Write-Host "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
Write-Host "Installing Ultralytics (YOLO) and dependencies..."
pip install ultralytics pyyaml pandas opencv-python
Write-Host "Environment setup complete."
