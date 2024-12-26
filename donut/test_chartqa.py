import argparse
from transformers import DonutProcessor, VisionEncoderDecoderModel
from transformers import VisionEncoderDecoderConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from data.chartqa import ChartQADatasetTest
import torch
from torch.nn.utils.rnn import pad_sequence
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

def compute_metric(pred, gt):
  try:
    gt = float(gt)
    pred = float(pred)
    return abs(gt - pred) / abs(gt) <= 0.05
  except:
    return str(gt).lower() == str(pred).lower()

def run_test(model, processor,test_loader, device,args):
    model.eval()  # Set model to evaluation mode
    model.to(device)
    total, correct = 0, 0
    predictions = []
    results = []
    scores_human = []
    scores_aug = []

    with torch.no_grad():
        for batch in test_loader:
            pixel_values, decoder_input_ids, prompt_end_idxs, answers, test_types = batch
            pixel_values = pixel_values.to(device)
            decoder_input_ids = decoder_input_ids.to(device)
            
            decoder_prompts = pad_sequence(
                [input_id[: end_idx + 1] for input_id, end_idx in zip(decoder_input_ids, prompt_end_idxs)],
                batch_first=True
            ).to(device)
            
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_prompts,
                max_length=args.max_length,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=4,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True
            )
          
            predictions = processor.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            predictions = [seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "") for seq in predictions]


            for pred, answer, test_type in zip(predictions, answers, test_types):
                pred = pred.split("<s_answer>")[1] 
                pred = pred.replace(processor.tokenizer.eos_token, "").replace("<s>", "").strip(' ')
                answer = answer.split("<s_answer>")[1] 
                answer = answer.replace(processor.tokenizer.eos_token, "").strip(' ')
                correct=compute_metric(answer, pred)
                score = 1 if correct else 0

                result = {"prediction": pred, "answer": answer, "correct": correct, "type": test_type}
                print(result)

                results.append(result)
                if test_type == "human":
                    scores_human.append(score)
                elif test_type == "augmented":
                    scores_aug.append(score)

        output_dir = args.output_dir 
        os.makedirs(output_dir, exist_ok=True)  

        human_file_path = os.path.join(output_dir, 'test_results_human.txt')
        augmented_file_path = os.path.join(output_dir, 'test_results_augmented.txt')

        with open(human_file_path, 'w') as file:
            for result in [res for res in results if res['type'] == 'human']:
                file.write(f"Prediction: {result['prediction']}, Answer: {result['answer']}, Correct: {result['correct']}\n")
            file.write(f"Human Accuracy: {sum(scores_human) / len(scores_human):.4f}\n")
            print(f"Human Accuracy: {sum(scores_human) / len(scores_human):.4f}\n")
        
        with open(augmented_file_path, 'w') as file:
            for result in [res for res in results if res['type'] == 'augmented']:
                file.write(f"Prediction: {result['prediction']}, Answer: {result['answer']}, Correct: {result['correct']}\n")
            file.write(f"Augmented Accuracy: {sum(scores_aug) / len(scores_aug):.4f}\n")
            print(f"Augmented Accuracy: {sum(scores_aug) / len(scores_aug):.4f}\n")

def main():
    parser = argparse.ArgumentParser(description='Test Chart Transformer')
    parser.add_argument('--data-path', type=str, default="ahmed-masry/chartqa_without_images", help='Path to the data file')
    parser.add_argument('--test-images', type=str, default='/content/ChartQA/ChartQA Dataset/test/png/', help='Path to the test images')
    parser.add_argument('--output-dir', type=str, default="/content/output_data", help='Path to the output folder')
    parser.add_argument('--checkpoint_processor', type=str, default= "naver-clova-ix/donut-base", help='Checkpoint path for processor')
    parser.add_argument('--checkpoint-model', type=str, default= "naver-clova-ix/donut-base", help='Checkpoint path for model')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum length for decoder generation')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of workers for data loading')


    args = parser.parse_args()

    processor = DonutProcessor.from_pretrained(args.checkpoint_processor)
    model = VisionEncoderDecoderModel.from_pretrained(args.checkpoint_model)
    
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    dataset = load_dataset(args.data_path)
    test_dataset = ChartQADatasetTest(dataset["test"], images_folder=args.test_images, processor=processor, max_length=args.max_length, split="test", prompt_end_token="<s_answer>", task_prefix="<chartqa>")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_test(model, processor,test_loader, device,args)




if __name__ == '__main__':
    main()
