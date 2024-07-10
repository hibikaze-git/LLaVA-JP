import json
import random
import re

from pathlib import Path

random.seed(42)


def check_abnormal_answer(answer):
    # アルファベット1文字だけの回答を除去
    pattern = re.compile(r"^[Ａ-Ｚａ-ｚ]$")

    if pattern.search(answer):
        return True
    else:
        return False


if __name__ == '__main__':
    dataset_dir = './dataset/v0'
    stair_captions_dir = 'VisualGenome'
    caption_path = Path(dataset_dir, stair_captions_dir, 'question_answers.json')
    image_path = Path(dataset_dir, stair_captions_dir, 'image_data.json')
    
    stair_llava_formats = []

    with caption_path.open('r', encoding='utf-8') as f:
        caption_data = f.read()
        caption_data_json = json.loads(caption_data)

    with image_path.open('r', encoding='utf-8') as f:
        image_data = f.read()
        image_data_json = json.loads(image_data)

    for image, annotation in zip(image_data_json, caption_data_json):
        #print(image)
        #print(annotation)

        for qas in annotation['qas']:
            llava_format = {}
            conversations = []

            short_answer_prompt = "1つの単語またはフレーズで回答してください。"
            question = qas["question"]

            #if check_abnormal_answer(qas["answer"]):
                #print(question)
                #print(qas["answer"])
                #continue

            delimiters = ["。", "?", "？", " ", "　"]

            if question[-1] in ["。", ".", "?", "？", "!", "！", " ", "　"]:
                question = question + short_answer_prompt
            else:
                question = question + random.choice(delimiters) + short_answer_prompt

            conversation_user = {
                'from': 'ユーザー',
                'value': f'{question}\n<image>'
            }
            conversation_system = {
                'from': 'システム',
                'value': qas["answer"]
            }
            conversations.append(conversation_user)
            conversations.append(conversation_system)

            llava_format['id'] = qas['qa_id']
            llava_format['image'] = f"{qas['image_id']}.jpg"
            llava_format['conversations'] = conversations

            #print(llava_format)

            stair_llava_formats.append(llava_format)

    print(len(stair_llava_formats))

    chat_ja_path = Path('./dataset/v0', 'llava_visual_genome_ja_after_short_prompt_curation.json')
    with open(chat_ja_path, mode="w") as f:
        json.dump(stair_llava_formats, f, indent=2, ensure_ascii=False)
    