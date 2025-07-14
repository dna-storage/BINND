import torch

def test_model(model, dataloader, device, log_dir):

    test_log = open(f'{log_dir}/test_log.csv', 'w')
    test_log.write("Label,Probability\n")

    model.eval()
    
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            inputs, labels = sample_batched['matrix'].to(device), sample_batched['label'].to(device)
            
            outputs = model(inputs.float())

            for i in range(len(labels)):
                test_log.write(f'{labels[i].item()},{outputs[i].item()}\n')
            
            test_log.flush()

            print(f"Batch {i_batch}/{len(dataloader)} completed.")

    test_log.close()


