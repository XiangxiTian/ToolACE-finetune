from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset


def format_conversation_to_messages(
		system: Optional[str],
		conversations: Optional[List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:

	at_least_one_assistant_message = False

	messages: List[Dict[str, str]] = []
	messages.append({"role": "system", "content": system.strip()})

	for turn in conversations:
		try:
			if 'from' in turn:
				role = turn["from"]
			else:
				role = turn["role"]
			role = role.strip().lower()
			if role == "assistant":
				at_least_one_assistant_message = True

			if 'value' in turn:
				content = turn["value"]
			else:
				content = turn["content"]
			content = content.strip()
   
			messages.append({"role": role, "content": content})
		except Exception as e:
			print(turn)
			print(f"Error formatting conversation: {e}")
			raise e

	if not at_least_one_assistant_message:
		return {"messages": None}

	return {"messages": messages}


def get_toolace_datasets(
		validation_fraction: float = 0.1,
		seed: int = 42,
) -> Tuple[Dataset, Dataset]:
	"""
	Load Team-ACE/ToolACE from Hugging Face and return (train_ds, val_ds),
	where each item is {'messages': [...]}. Invalid rows are dropped.
	"""
	# Try to load a single split for consistency, otherwise merge all available.
	raw = load_dataset("Team-ACE/ToolACE", split="train")
	print(f"Raw dataset loaded: {len(raw)} samples")

	def _map_example(example: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Map a raw ToolACE example into {'messages': [...]} for SFTTrainer.
		"""
		msg_obj = format_conversation_to_messages(
			example["system"],
			example["conversations"],
		)
		return msg_obj

	mapped: Dataset = raw.map(
		_map_example,
		remove_columns=raw.column_names,
	)
	print(f"After mapping: {len(mapped)} samples")

	# Filter out rows where mapping failed
	mapped = mapped.filter(lambda ex: ex["messages"] is not None)
	print(f"After filtering: {len(mapped)} samples")

	# Create train/validation split.
	split: DatasetDict = mapped.train_test_split(
		test_size=validation_fraction,
		seed=seed,
	)
	train_ds: Dataset = split["train"]
	val_ds: Dataset = split["test"]

	print(f"Training samples: {len(train_ds)}")
	print(f"Validation samples: {len(val_ds)}")
	print(f"Total samples: {len(train_ds) + len(val_ds)}")

	return train_ds, val_ds


