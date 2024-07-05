import json
from pathlib import Path


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

            conversation_user = {
                'from': 'ユーザー',
                'value': f'{"1つの単語またはフレーズで回答してください。" + qas["question"]}\n<image>'
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

    chat_ja_path = Path('./dataset/v0', 'llava_visual_genome_ja.json')
    with open(chat_ja_path, mode="w") as f:
        json.dump(stair_llava_formats, f, indent=2, ensure_ascii=False)
    