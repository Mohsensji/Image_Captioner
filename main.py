
from architecture import CNNEncoder,LSTMDecoder
from preprocessing import preprocess_captions
from trainer import ImageCaptioningTrainer
from dataset import CocoDataset
def main():
    train_img_folder = '/kaggle/input/coco-2017-dataset/coco2017/train2017'
    train_ann_file = '/kaggle/input/coco-2017-dataset/coco2017/annotations/captions_train2017.json'
    val_img_folder = '/kaggle/input/coco-2017-dataset/coco2017/val2017'
    val_ann_file = '/kaggle/input/coco-2017-dataset/coco2017/annotations/captions_val2017.json'

    train_dataset = CocoDataset(train_img_folder, train_ann_file)
    val_dataset = CocoDataset(val_img_folder, val_ann_file)

    train_annotations = train_dataset.img_id_to_ann.values()
    vocab = preprocess_captions(train_annotations)
    
    encoder = CNNEncoder()
    decoder = LSTMDecoder(
        input_size=256,  
        hidden_size=512,
        output_size=len(vocab),
        num_layers=1,
        encoder=encoder
    )

    trainer = ImageCaptioningTrainer(
        encoder=encoder,
        decoder=decoder,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        vocab=vocab
    )

    train_losses, val_losses = trainer.train(num_epochs=20)

if __name__ == "__main__":
    main()