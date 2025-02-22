from transformers import AutoTokenizer, AutoModelForQuestionAnswering, DefaultDataCollator, Trainer, TrainingArguments, pipeline
from datasets import load_dataset
from tqdm import tqdm

# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-squad")
model = AutoModelForQuestionAnswering.from_pretrained("savasy/bert-base-turkish-squad")

def normalize_text(text):
    # Lowercase and remove punctuation.
    text = text.lower()
    text = ''.join(ch for ch in text if ch.isalnum() or ch.isspace())
    return ' '.join(text.split())

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    if len(common_tokens) == 0:
        return 0
    
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def evaluate_qa_model(model, tokenizer, eval_dataset):
    total_em = 0
    total_f1 = 0
    count = 0
    
    question_answerer = pipeline("question-answering", model=model, tokenizer=tokenizer)
    
    for example in tqdm(eval_dataset):
        pred = question_answerer(
            question=example['question'],
            context=example['context']
        )
        
        true_answer = example['answers']['text'][0]
        pred_answer = pred['answer']
        
        em_score = compute_exact_match(pred_answer, true_answer)
        f1_score = compute_f1(pred_answer, true_answer)
        
        total_em += em_score
        total_f1 += f1_score
        count += 1
    
    final_em = total_em / count * 100
    final_f1 = total_f1 / count * 100
    
    return {
        'exact_match': final_em,
        'f1': final_f1
    }

def preprocess(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def main():
    # Load and preprocess dataset
    mydataset = load_dataset("json", data_files="data/dataset_squad.json", field="data")
    mydataset = mydataset["train"].train_test_split(test_size=0.2)
    tokenized_dataset = mydataset.map(preprocess, batched=True)

    # Training setup
    data_collator = DefaultDataCollator()
    training_args = TrainingArguments(
        output_dir="gtu_qa_model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train and save model
    trainer.train()
    model.save_pretrained("gtu_qa_model")
    tokenizer.save_pretrained("gtu_qa_model")

    # Evaluate model
    print("Evaluating model...")
    eval_results = evaluate_qa_model(model, tokenizer, tokenized_dataset["test"])
    print(f"Evaluation Results:")
    print(f"Exact Match: {eval_results['exact_match']:.2f}%")
    print(f"F1 Score: {eval_results['f1']:.2f}%")

    # Test example
    question_answerer = pipeline("question-answering", model="gtu_qa_model", tokenizer="gtu_qa_model")
    question = "Kayıt yenileme ne sıklıkla yapılmalıdır?"
    context = "Kayıt yenilenmesi MADDE 10 – (1) Öğrenciler her yarıyıl, akademik takvimde belirlenen tarihlerde Senato esasları ile belirlenen usullere göre kayıtlarını yenilemek zorundadır. Kayıtlarını yenileyemeyen öğrenciler akademik takvimde belirtilen mazeretli kayıt süresi içerisinde ilgili fakülte öğrencileri için dekanlığa, ön lisans öğrencileri için meslek yüksekokulu müdürlüğüne başvururlar. Mazeretleri geçerli görülen öğrenciler, kayıtlarını ders ekleme–çıkarma süresinin sonuna kadar yaptırmakla yükümlüdürler. Ders kaydı yaptırılmayan yarıyıl, öğretim süresinden sayılır."
    print("\nTest Example:")
    print("- Question: ", question)
    print("- Context: ", context.split('MADDE')[0].strip())
    print("- Answer: ", question_answerer(question=question, context=context))

if __name__ == "__main__":
    main()