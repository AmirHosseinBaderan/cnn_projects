import torch


def validate(
    model,
    dataloader,
    criterion,
    device,
    metrics
):
    model.eval()

    total_loss = 0.0
    prediction_results = None

    with torch.no_grad():

        for batch_idx, batch in enumerate(dataloader):

            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            output = model(images)

            loss = criterion(
                logits=output["logits"],
                targets=targets,
                input_lengths=output["input_lengths"],
                target_lengths=target_lengths
            )

            total_loss += loss.item()

            if batch_idx == 0:

                prediction_results = metrics.evaluate_batch(
                    logits=output["logits"],
                    labels=batch["labels"]
                )

    return {
        "loss": total_loss / len(dataloader),
        "predictions": prediction_results
    }