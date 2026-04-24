import numpy as np
import os
import json
from sklearn.metrics import f1_score

class Evaluator:
    def __init__(self, field):
        self.field = field

    def evaluate(self, pred_record_folder):
        files = sorted(os.listdir(pred_record_folder))
        all_labels = []
        all_predictions = []
        
        for file in files:
            '''
            Source code before update:
            '''
            # true_label = 1 if file.startswith("Glaucoma") else 0
            # all_labels.append(true_label)
            
            # pred_file_path = os.path.join(pred_record_folder, file)
            # with open(pred_file_path, 'r', encoding='utf-8') as f:
            #     result = json.load(f)
            # prediction = result[self.field]
            # if "yes" in prediction.lower():
            #     all_predictions.append(1)
            # elif "no" in prediction.lower():
            #     all_predictions.append(0)
            # else:
            #     all_predictions.append(0)
            #    # Skip if not a directory (could be file)
            case_dir = os.path.join(pred_record_folder, file)
            if not os.path.isdir(case_dir):
                continue
        
            true_label = 1 if file.startswith("Glaucoma") else 0
            all_labels.append(true_label)
        
            # Read final_diagnosis.json from the case directory
            final_diag_path = os.path.join(case_dir, 'final_diagnosis.json')
            if not os.path.exists(final_diag_path):
                print(f"Warning: {final_diag_path} not found, skipping")
                all_predictions.append(0)
                continue
        
            with open(final_diag_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
        
            # Extract diagnosis from nested structure
            diagnosis_str = result.get('overall', {}).get('diagnosis', 'Negative')
        
            # Convert "Positive"/"Negative" to 1/0
            if diagnosis_str.lower() == "positive":
                all_predictions.append(1)
            else:
                all_predictions.append(0)
        pos_count = all_labels.count(1)
        neg_count = all_labels.count(0)
        
        all_labels_np = np.array(all_labels)
        all_predictions_np = np.array(all_predictions)
        
        pos_correct = np.sum((all_labels_np == 1) & (all_predictions_np == 1))
        neg_correct = np.sum((all_labels_np == 0) & (all_predictions_np == 0))
        
        mean_acc = (pos_correct / pos_count + neg_correct / neg_count) / 2 if pos_count and neg_count else 0
        f1 = f1_score(all_labels, all_predictions)
        
        # print the results
        print(f"Glaucoma Accuracy: {pos_correct}")
        print(f"Non-Glaucoma Accuracy: {neg_correct}")
        print(f"Mean Accuracy: {mean_acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return {
            "Glaucoma Accuracy": pos_correct,
            "Non-Glaucoma Accuracy": neg_correct,
            "Mean Accuracy": mean_acc,
            "F1 Score": f1
        }


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_pred_record_folder = os.path.join(current_dir, "Glaucoma", "record")

    evaluator = Evaluator(field="overall.diagnosis")
    evaluator.evaluate(default_pred_record_folder)