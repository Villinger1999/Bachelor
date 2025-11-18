"""flower-tutorial: A Flower / PyTorch app."""

import torch
from torchvision.utils import save_image
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from flower_tutorial.task import Net, load_data
from flower_tutorial.task import test as test_fn
from flower_tutorial.task import train as train_fn
from flower_tutorial.task import attack as attack_fn

# Flower ClientApp
app = ClientApp()


# @app.train()
# def train(msg: Message, context: Context):
#     """Train the model on local data."""

#     # Load the model and initialize it with the received weights
#     model = Net()
#     model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     # Load the data
#     partition_id = context.node_config["partition-id"]
#     num_partitions = context.node_config["num-partitions"]
#     trainloader, _ = load_data(partition_id, num_partitions)

#     # Call the training function
#     train_loss = train_fn(
#         model,
#         trainloader,
#         context.run_config["local-epochs"],
#         msg.content["config"]["lr"],
#         device,
#     )

#     # Construct and return reply Message
#     model_record = ArrayRecord(model.state_dict())
#     metrics = {
#         "train_loss": train_loss,
#         "num-examples": len(trainloader.dataset),
#     }
#     metric_record = MetricRecord(metrics)
#     content = RecordDict({"arrays": model_record, "metrics": metric_record})
#     return Message(content=content, reply_to=msg)

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data and (optionally) run iDLG attack."""
    import os

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _ = load_data(partition_id, num_partitions)

    # Train and get leaked gradients for first batch
    train_loss, leaked_grads, real_img = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
        return_leaked_grads=True,
    )

    # Optional: run the iDLG attack here
    modelpre = Net()
    modelpre.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    modelpre.to(device)

    reconstructionAndLabel = attack_fn(
        modelpre,
        leaked_grads,
        (2, 3, 32, 32),  # CIFAR-10
        device,
        train_ite=200,   # LBFGS outer iterations
        clamp=(-1.0, 1.0),
    )

    os.makedirs("reconstructions", exist_ok=True)
    img = reconstructionAndLabel[0].detach().cpu()
    img = (img * 0.5) + 0.5
    img = img.clamp(0, 1)
    save_image(img, "reconstructions/flowerreconstruction.png")

    real_img_to_save = (real_img * 0.5 + 0.5).clamp(0, 1)
    save_image(real_img_to_save, "reconstructions/original_training_image.png")

    # Construct and return reply Message (standard FedAvg)
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valloader = load_data(partition_id, num_partitions)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)

# @app.attack()
# def attack(msg: Message, context: Context):
#     import copy
#     import os

#     # initial weights from message
#     initial_state = copy.deepcopy(msg.content["arrays"].to_torch_state_dict())
#     # Load the model and initialize it with the received weights
#     model = Net()
#     model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     # Load the data
#     partition_id = context.node_config["partition-id"]
#     num_partitions = context.node_config["num-partitions"]
#     trainloader, _ = load_data(partition_id, num_partitions)

#     # Call the training function
#     train_loss, leaked_grads = train_fn(model, trainloader, context.run_config["local-epochs"], msg.content["config"]["lr"], device, return_leaked_grads=True)
    
#     # Pre-update model (for attack)
#     modelpre = Net()
#     modelpre.load_state_dict(initial_state)
#     modelpre.to(device)
    
#     # Call attack function
    
#     reconstructionAndLabel = attack_fn(
#         modelpre,
#         leaked_grads, 
#         (1,3,224,224), 
#         device,
#         learning_rate=msg.content["config"]["lr"]
#         )
    
#     os.makedirs("reconstructions", exist_ok=True)
#     img = reconstructionAndLabel[0].detach().cpu()
#     save_image(img, "reconstructions/flowerreconstruction.png")
    
#     # Construct and return reply Message    
#     model_record = ArrayRecord(model.state_dict())
#     metrics = {
#         "train_loss": train_loss,
#         "num-examples": len(trainloader.dataset),
#     }
#     metric_record = MetricRecord(metrics)
#     content = RecordDict({"arrays": model_record, "metrics": metric_record})
    
#     return Message(content=content, reply_to=msg)