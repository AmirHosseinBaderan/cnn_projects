from tqdm import tqdm


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0

    progress_bar = tqdm(
        dataloader,
        desc="Train",
        leave=False,
        dynamic_ncols=True,
        unit="batch",
    )

    for batch_idx, batch in enumerate(progress_bar, start=1):

        images = batch["images"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        optimizer.zero_grad(set_to_none=True)

        output = model(images)

        loss = criterion(
            logits=output["logits"],
            targets=targets,
            input_lengths=output["input_lengths"],
            target_lengths=target_lengths,
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        average_loss = total_loss / batch_idx

        progress_bar.set_postfix(
            loss=f"{average_loss:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    return total_loss / len(dataloader)