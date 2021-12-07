conda create -n 3d_dfa_v2 python=3.8 -y;
conda activate 3d_dfa_v2;

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y;

pip install --upgrade pip;
pip install -r requirements.txt;

sh ./build.sh;
