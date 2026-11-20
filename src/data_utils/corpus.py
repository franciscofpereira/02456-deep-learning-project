import os
import shutil
from datasets import Dataset, DatasetDict, load_from_disk
from src.data_utils.download_retrieval_datasets import get_fiqa_dataset, get_squad_dataset
import json

class Corpus:
    def __init__(self, dataset_name: str, base_dir: str):
        self.dataset_name = dataset_name
        if not os.path.exists(f'{base_dir}/{self.dataset_name}'):
            raw_data_dir = f'{base_dir}/raw'
            os.makedirs(raw_data_dir, exist_ok=True)
            # load and process data
            self.ds = self.__load_data(base_dir=raw_data_dir)
            self.__rename_columns()
            if self.dataset_name == 'squad':
                self.__prepare_squad()
            self.relevant_ids = self.__assign_ids()
            os.makedirs(f'{base_dir}/{self.dataset_name}', exist_ok=True)
            DatasetDict(self.ds).save_to_disk(f'{base_dir}/{self.dataset_name}')
            with open(f'{base_dir}/{self.dataset_name}/relevant_ids.json', 'w') as f:  # add
                json.dump(self.relevant_ids, f)  
            print(f'Saved {self.dataset_name} dataset to path {base_dir}/{self.dataset_name}.')
            shutil.rmtree(f'{base_dir}/raw')
        else:
            self.ds = load_from_disk(f'{base_dir}/{self.dataset_name}')
            with open(f'{base_dir}/{self.dataset_name}/relevant_ids.json') as f:  # add
                self.relevant_ids = json.load(f)


    @property
    def train(self):
        return self.ds.get('train', None)
    
    @property
    def val(self):
        return self.ds.get('validation', None)

    @property
    def test(self):
        return self.ds.get('test', None)

    @property
    def all_documents(self):
        train_text, train_ids = list(self.train['text']), list(self.train['document_id'])
        val_text, val_ids =     list(self.val['text']), list(self.val['document_id'])

        all_text = train_text + val_text
        all_ids = train_ids   + val_ids

        if self.test is not None:
            test_text, test_ids = list(self.test['text']), list(self.test['document_id'])
            all_text += test_text
            all_ids  += test_ids

        # SQuAD: several questions share the same paragraph (same document id).
        # Keep every document only once, otherwise the corpus contains duplicates.
        # (FiQA has unique ids, so this branch is skipped and nothing changes.)
        flat = [(t, i) for texts, ids in zip(all_text, all_ids) for t, i in zip(texts, ids)]
        if len({i for _, i in flat}) != len(flat):
            unique = {}
            for t, i in flat:
                unique.setdefault(i, t)
            return Dataset.from_dict({'text': [[t] for t in unique.values()],
                                      'document_id': [[i] for i in unique.keys()]})

        return Dataset.from_dict({'text': all_text,
                                  'document_id': all_ids})


    def __load_data(self, base_dir) -> dict[Dataset]:
        assert self.dataset_name in ['squad', 'fiqa'], f'Dataset {self.dataset_name} not supported. Options are ["squad", "fiqa"]'
        
        if self.dataset_name == 'fiqa':
            ds = get_fiqa_dataset(base_dir)

        elif self.dataset_name == 'squad':
            ds = get_squad_dataset(base_dir)

        return {key: ds[key] for key in ds.keys()}

    def __rename_columns(self) -> dict[Dataset]:
        column_mapping = {'fiqa':  'ground_truths',
                          'squad': 'context'
                         }
        
        for key in self.ds:
            self.ds[key] = self.ds[key].rename_column(column_mapping[self.dataset_name], 'text')
        return self.ds

    def __prepare_squad(self):
        """Bring SQuAD into the same format as FiQA.

        - 'text' becomes a list with one relevant document (the paragraph), like FiQA's list of ground truths.
        - SQuAD has no test split: the official validation split is divided 50/50 into validation and test
          (fixed seed, so every run uses the same split).
        """
        for key in self.ds:
            self.ds[key] = self.ds[key].map(lambda ex: {'text': [ex['text']]})

        split = self.ds['validation'].train_test_split(test_size=0.5, seed=42)
        self.ds['validation'] = split['train']
        self.ds['test'] = split['test']

    def __assign_ids(self):
        q_id_counter = 0
        doc_id_counter = 0
        relevant_document_ids = {}
        # SQuAD: the same paragraph gets the same id for every question that is asked about it
        doc_ids_by_text = {} if self.dataset_name == 'squad' else None

        for key in self.ds:
            query_ids = []
            document_ids = []
            texts = self.ds[key]['text']
            for i in range(len(self.ds[key])):
                query_ids.append(f'q{q_id_counter}')
                documents = texts[i]
                relevant_ids = []
                for doc in documents:
                    if doc_ids_by_text is not None:
                        if doc not in doc_ids_by_text:
                            doc_ids_by_text[doc] = f'doc{doc_id_counter}'
                            doc_id_counter += 1
                        relevant_ids.append(doc_ids_by_text[doc])
                        continue
                    relevant_ids.append(f'doc{doc_id_counter}')
                    doc_id_counter += 1
                
                document_ids.append(relevant_ids)
                
                relevant_document_ids[f'q{q_id_counter}'] = relevant_ids
                q_id_counter += 1
                
            self.ds[key] = self.ds[key].add_column('query_id', query_ids)
            self.ds[key] = self.ds[key].add_column('document_id', document_ids)

        return relevant_document_ids


    def get_batch(self, batch_size: int, split: str = 'train', shuffle: bool = True):
        '''
        Get a batch of data from the specified split.
        '''
        assert split in ['train', 'val', 'test'], f'Split {split} not supported. Options are ["train", "val", "test"]'
        dataset = self.ds[split]
        if shuffle:
            dataset = dataset.shuffle()
        
        for i in range(0, len(dataset), batch_size):
            yield dataset[i:i + batch_size]

    def get_length(self, split: str = 'train'):
        '''
        Get the length of the specified split.
        '''
        assert split in ['train', 'val', 'test'], f'Split {split} not supported. Options are ["train", "val", "test"]'
        return len(self.ds[split])

        
