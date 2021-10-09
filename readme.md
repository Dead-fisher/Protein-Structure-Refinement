# refinement

在RiD环境下运行：
`python mk_rid_refinement.py -h`
查看帮助

## 参数说明

`TASK` 任务名称

`--mol` 用于指定初始的PDB文件。

`'-c', '--dihedral-index'` 用于指定作为CV的二面角的序号。

`--all-dihedral` 使用后将默认选择所有的二面角作为CV。

`'-d', '--bottom-width'` 用于设置平底势的宽度。

`'-r', '--config'` 通过JSON文件设置。

## 样例

```bash
cd /home/dongdong/wyz/refinement3_2chain
python mk_rid_refinement.py 4m6o --mol /home/dongdong/wyz/refinement3_2chain/PDB/4m6o --all-dihedral

cd /home/dongdong/wyz/refinement3_2chain/4m6o.run06
python rid.py rid.json
```


