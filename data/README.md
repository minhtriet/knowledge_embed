Data
----------------------------
`gradgraph` comes with two standard datasets: 

- **FB15K**:  14,951 entities and 1,345 relationships, with 592,213 triplets:
  - 483,142 train
  - 50,000 validation
  - 59,071 test
- **WN18**: 40,943 entities and 18 relationships, with 151442 triplets:
  -  141,442 train
  - 5,000 validation
  - 5,000 test

Plus FB15K-237, which is a subset of FB15K. We use the same Train/Validation/Test splits introduced in [Bordes et. al. 2011 & 2013](https://everest.hds.utc.fr/doku.php?id=en:smemlj12).

New datasets can be added, but must have the following formats. `\t` denotes a tab, and `ID` denotes the embedding index for the given object.

- `entity2id.txt`: Mapping from entity names to embedding indices. First line is the number of entities, followed by `ename\tID`.
- `relation2id.txt`: Mapping from relation names to embedding indices. First line is the number of relations, followed by `rname\tID`. 
- `train2id.txt`, `valid2id.txt`, and `test2id.txt`: The training, validation and test splits. The first line is the number of triplets, followed by `entity_1ID`, `entity_2ID`, and `relationID` separated by spaces, in that order.
- `type_constrain.txt` (optional): A file imposing type constraints on each relation. The first line is the number of relations. Each of the following lines has the format `relationID\tn\te1ID e2ID e3ID...` with tab-separated fields. The three fields are: (1) the relation index `relationID`, (2) the number `n` of entities that can occur as a left or right argument of the relation, and (3) the indices of all entities that can occur as a right or left argument of the relation, separated by spaces. Each relation has two lines. The first gives the entities that can occur in the right position for the relation, and the second gives those that can occur in the left position.

For the purposes of model analysis, there are additional mappings `name2idx` and `idx2name` between human-readable names embedding embedding indices that allow users to probe the model with specific queries (see main README for details). For FB15K, we also provide versions of the training and test sets with types assigned to left entities. In Freebase, entities occurring in particular relationships are assigned *types*, capturing the intuition that, for instance, `Stan Lee` is considered as an instance of the `writer` type when considered as the author of the `Spiderman` comics, and an instance of the `actor` type when appearing in one of the `Spiderman` films. These type assignments can be used as an additional training resource for adding the structure of the Freebase ontology to the learned semantic space. The typed training/validation/test files have the same format as the untyped ones, but with the type of the left entity appended to line---i.e. each line is `entity1ID entity2ID relationID typeID`. 


