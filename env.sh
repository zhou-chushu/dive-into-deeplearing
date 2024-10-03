conda create -n cszhou python=3.9
conda activate cszhou

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host https://pypi.tuna.tsinghua.edu.cn

pip install torch
pip install transformers
pip install datasets
pip install evaluate
pip install peft
pip install scikit-learn
pip install tqdm
pip install tensorboard
pip install ipywidgets
pip install termcolor
pip install fschat
pip install psutil
pip install accelerate
pip install bitsandbytes
pip install sentencepiece
pip install kaleido

pip install matplotlib
pip install plotly
pip install nbformat
pip install transformer_lens
pip install pytest
pip install ipykernel