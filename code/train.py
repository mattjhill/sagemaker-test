import argparse
import json
import os
import tensorflow as tf
from transformer_instacart import create_decoder_model

def main(args):
    print("running model")
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    print(os.listdir(args.train))
    # print('loading data')
    # products = pandas.read_csv('data/products.csv')
    # product_id_max = products['product_id'].max()

    # orders_meta = pandas.read_csv('data/orders.csv', index_col='order_id')
    # orders_meta = orders_meta[orders_meta['eval_set'] == 'prior']
    # orders = pandas.read_csv('data/order_products__prior.csv')
    # orders_meta['days_since_first_order'] = orders_meta.groupby('user_id')[['days_since_prior_order']].cumsum()
    # orders_meta['days_since_first_order'] = orders_meta['days_since_first_order'].fillna(-1)

    # orders_meta[['order_dow', 'order_hour_of_day', 'days_since_first_order']] += 1
    # orders = orders.join(orders_meta, on='order_id')
    # orders = orders.sort_values(by=['user_id', 'order_number', 'add_to_cart_order'])
    
    # first_orders = orders.groupby('order_id', as_index=False).first()
    # first_orders['add_to_cart_order'] = 0
    # first_orders['product_id'] = product_id_max + 1
    # product_id_max += 1

    # last_orders = orders.groupby('order_id', as_index=False).last()
    # last_orders['add_to_cart_order'] += 1
    # last_orders['product_id'] = product_id_max + 1
    # product_id_max += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--hosts',type=list,default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host',type=str,default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--train',type=str,required=False,default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--validation',type=str,required=False,default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--eval',type=str,required=False,default=os.environ.get('SM_CHANNEL_EVAL'))
    parser.add_argument('--model_dir',type=str,required=True,help='The directory where the model will be stored.')
    parser.add_argument('--model_output_dir',type=str,default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output_data_dir',type=str,default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--output-dir',type=str,default=os.environ.get('SM_OUTPUT_DIR'))
    parser.add_argument('--tensorboard-dir',type=str,default=os.environ.get('SM_MODULE_DIR'))
    parser.add_argument('--weight-decay',type=float,default=2e-4,help='Weight decay for convolutions.')
    parser.add_argument('--learning-rate',type=float,default=0.001,help='Initial learning rate.')
    parser.add_argument('--data-config',type=json.loads,default=os.environ.get('SM_INPUT_DATA_CONFIG'))
    parser.add_argument('--fw-params',type=json.loads,default=os.environ.get('SM_FRAMEWORK_PARAMS'))
    parser.add_argument('--optimizer',type=str,default='adam')
    parser.add_argument('--momentum',type=float,default='0.9')
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dff', type=int, default=256)

    args = parser.parse_args()

    main(args)
