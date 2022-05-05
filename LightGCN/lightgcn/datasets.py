import os

import pandas as pd
import torch


def prepare_dataset(device, basepath, verbose=True, logger=None):
    """train과 test 데이터셋을 만들어주는 함수

    Args:
        device (str): cpu 또는 cuda:0 선택
        basepath (str): 데이터셋이 저장된 폴더 경로
        verbose (bool, optional): 데이터셋 정보를 출력할 것인지. Defaults to True.
        logger (object, optional): 데이터셋 정보를 출력할 logger. Defaults to None.

    Returns:
        train_data_proc (dict) : train_data에 대한 edge, label 생성
        test_data_proc (dict) : test_data에 대한 edge, label 생성
        len(id2index) (dict) : 모든 interaction의 수
    """

    # basepath로 부터 train_data.csv + test_data.csv 합친 것 불러오기
    data = load_data(basepath)

    # 불러온 data를 answerCode를 기준으로 다시 train과 test로 분리
    train_data, test_data = separate_data(data)

    # user2index, assessmentItemID2index 계산
    id2index = indexing_data(data)

    # Graph 정보 생성: dict(Edge, Label)
    # - Edge : userID <----> assessmentItemID
    # - Label : answerCode
    train_data_proc = process_data(train_data, id2index, device)
    test_data_proc = process_data(test_data, id2index, device)

    if verbose:
        print_data_stat(train_data, "Train", logger=logger)
        print_data_stat(test_data, "Test", logger=logger)

    return train_data_proc, test_data_proc, len(id2index)


def load_data(basepath):
    """
    train과 test 데이터셋을 불러와서 합친 후,
    userID와 assessmentItemID 쌍이 고유한 것들만 남기고
    나머지는 제거한다.

    Args:
        basepath (str): 데이터셋이 존재하는 폴더 경로

    Returns:
        data (pd.DataFrame): train과 test set이 합쳐진 데이터셋
    """
    path1 = os.path.join(basepath, "train_data.csv")
    path2 = os.path.join(basepath, "test_data.csv")
    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)

    data = pd.concat([data1, data2])
    data.drop_duplicates(
        subset=["userID", "assessmentItemID"], keep="last", inplace=True
    )

    return data


def separate_data(data):
    """
    data를 train과 test data로 재분리
    - answerCode >= 0  : train data
    - answerCode == -1 : test data

    Args:
        data (pd.DataFrame): train+test data

    Returns:
        train_data (pd.DataFrame)
        test_data (pd.DataFrame)
    """

    train_data = data[data.answerCode >= 0]
    test_data = data[data.answerCode < 0]

    return train_data, test_data


def indexing_data(data):
    """
    user를 0부터 다시 번호 매기기 (0 ~ n_user - 1)
    assessmentItemID 다시 번호 매기기 (0 ~ n_item - 1)

    Args:
        data (pd.DataFrame): train+test data

    Returns:
        id_2_index (dict): userID 또는 assessmentItemID를 reindexing 한 table
    """
    userid, itemid = (
        sorted(list(set(data.userID))),
        sorted(list(set(data.assessmentItemID))),
    )
    n_user, n_item = len(userid), len(itemid)

    userid_2_index = {v: i for i, v in enumerate(userid)}
    itemid_2_index = {v: i + n_user for i, v in enumerate(itemid)}
    id_2_index = dict(userid_2_index, **itemid_2_index)

    return id_2_index


def process_data(data, id_2_index, device):
    """
    user와 item을 edge로 연결하고, edge에 대한 정답을 label로 지정

    Args:
        data (pd.DataFrame): train+test data
        id_2_index (dict): userID 또는 assessmentItemID를 reindexing 한 table
        device (str): cpu 또는 cuda:0 선택

    Returns:
        dict(list(), list()): edge와 label list
    """
    edge, label = [], []
    for user, item, acode in zip(data.userID, data.assessmentItemID, data.answerCode):
        uid, iid = id_2_index[user], id_2_index[item]
        edge.append([uid, iid])
        label.append(acode)

    edge = torch.LongTensor(edge).T
    label = torch.LongTensor(label)

    return dict(edge=edge.to(device), label=label.to(device))


def print_data_stat(data, name, logger):
    userid, itemid = list(set(data.userID)), list(set(data.assessmentItemID))
    n_user, n_item = len(userid), len(itemid)

    logger.info(f"{name} Dataset Info")
    logger.info(f" * Num. Users    : {n_user}")
    logger.info(f" * Max. UserID   : {max(userid)}")
    logger.info(f" * Num. Items    : {n_item}")
    logger.info(f" * Num. Records  : {len(data)}")
