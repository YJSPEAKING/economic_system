import torch


def freeze_td3_actor(td3):
    td3.actor.eval()
    td3.actor_target.eval()
    td3.var_init = 0.0
    td3.var_stable = 0.0
    td3.var = 0.0
    td3.eval_noise_scale = 0.0
    for param in td3.actor.parameters():
        param.requires_grad = False
    for param in td3.actor_target.parameters():
        param.requires_grad = False


def load_actor_into_td3(td3, actor_state_dict, freeze=True):
    td3.actor.load_state_dict(actor_state_dict)
    td3.actor_target.load_state_dict(actor_state_dict)
    if freeze:
        freeze_td3_actor(td3)


def load_background_actors(checkpoint_path, consumption_agent=None, bank_agent=None, map_location="cpu", freeze=True):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    actors = checkpoint.get("actors", {})
    if consumption_agent is not None and "consumption1" in actors:
        load_actor_into_td3(
            consumption_agent.enterprise,
            actors["consumption1"]["actor_state_dict"],
            freeze=freeze,
        )
    if bank_agent is not None and "bank1" in actors:
        load_actor_into_td3(
            bank_agent.bank,
            actors["bank1"]["actor_state_dict"],
            freeze=freeze,
        )
    return checkpoint
