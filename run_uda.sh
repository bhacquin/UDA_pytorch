#pip uninstall tensorflow
#pip uninstall tensorboard
#pip install -q --ignore-installed tf-nightly-2.0-preview
#pip install tensorflow
#git clone https://github.com/bhacquin/pytorch_bert_addons.git
#pip install pytorch_bert_addons/pytorch-pretrained-BERT/.

#pip install tensorboardX
#pwd
rm -r logs
mkdir -p logs
tensorboard --host 0.0.0.0 --logdir=logs  &
python run_uda.py --sequence_length 256 --multi_gpu  
