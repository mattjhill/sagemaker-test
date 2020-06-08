import sagemaker
from sagemaker.tensorflow import TensorFlow
import boto3

session = sagemaker.Session(boto_session=boto3.Session(profile_name='knovita'))

estimator = TensorFlow(
    sagemaker_session=session,
    git_config={
        'repo': 'https://github.com/mattjhill/sagemaker-test.git',
        'branch': 'master'
    },
    source_dir='code',
    role='arn:aws:iam::712135249233:role/SageMakerRole',
    entry_point='train.py',
    framework_version='2.1.0',
    py_version='py3',
#    train_use_spot_instances=True,
#    train_max_wait=300,
    train_instance_count=1,
    train_instance_type='ml.p3.8xlarge',
)

estimator.fit('s3://knovita-sagemaker/instacart/', wait=True)
