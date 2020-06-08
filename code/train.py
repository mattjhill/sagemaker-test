import argparse
import json
import os
import tensorflow as tf

def main(args):
    print("running model")
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--hosts',type=list,default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host',type=str,default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--train',type=str,required=False,default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation',type=str,required=False,default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--eval',type=str,required=False,default=os.environ.get('SM_CHANNEL_EVAL'))
    parser.add_argument('--model_dir',type=str,required=True,help='The directory where the model will be stored.')
    parser.add_argument('--model_output_dir',type=str,default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output_data_dir',type=str,default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--output-dir',type=str,default=os.environ.get('SM_OUTPUT_DIR'))
    parser.add_argument('--tensorboard-dir',type=str,default=os.environ.get('SM_MODULE_DIR'))
    parser.add_argument('--weight-decay',type=float,default=2e-4,help='Weight decay for convolutions.')
    parser.add_argument('--learning-rate',type=float,default=0.001,help='Initial learning rate.')
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--data-config',type=json.loads,default=os.environ.get('SM_INPUT_DATA_CONFIG'))
    parser.add_argument('--fw-params',type=json.loads,default=os.environ.get('SM_FRAMEWORK_PARAMS'))
    parser.add_argument('--optimizer',type=str,default='adam')
    parser.add_argument('--momentum',type=float,default='0.9')
    
    args = parser.parse_args()

    main(args)
