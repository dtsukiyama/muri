
gcloud compute instances create $1 \
        --zone=us-west1-b \
        --image-family=chainer-latest-cu92-experimental\
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --accelerator='type=nvidia-tesla-k80,count=1' \
        --machine-type=n1-standard-8 \
        --boot-disk-size=120GB \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --metadata='install-nvidia-driver=True,proxy-mode=project_editors'
