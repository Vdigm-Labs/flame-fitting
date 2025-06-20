"""fit_lmk3d_ko_comment.py (주석 확장 버전)
────────────────────────────────────────────────────────────────────────────
이 파일은 Rubikplayer/flame‑fitting 저장소의 `fit_lmk3d.py` 를 기반으로 한다.
원본 코드의 구조는 유지하되, **학습 목적**으로 세세하고 장문의 한국어 주석을
넣어 두었다. 실제 프로덕션 코드에서는 이렇게 장황한 주석을 달지 않지만,
● 각 단계가 어떤 역할을 하는지
● 입력 데이터가 어떤 포맷이어야 하는지
● 파라미터가 어떤 물리적 의미를 가지는지
를 이해하는 데 도움을 주기 위해 상세 주석을 추가하였다.
────────────────────────────────────────────────────────────────────────────
필요 라이브러리
──────────────
- numpy       : 수치 연산 (파라미터 벡터, 버텍스 좌표 등)
- chumpy      : 자동 미분 & 최적화를 쉽게 해주는 라이브러리 (NumPy + autodiff)
- smpl_webuser: FLAME/SMPL 모델을 로딩하기 위한 유틸
- fitting.*   : Rubikplayer repo 내부의 보조 함수들 (임베딩, OBJ 저장 등)
────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import chumpy as ch
from os.path import join

# 🔽 FLAME 관련 로딩/유틸 함수 ----------------------
from smpl_webuser.serialization import load_model  # .pkl → chumpy 객체
from fitting.landmarks import load_embedding, landmark_error_3d  # 랜드마크 매핑 & 에러 함수
from fitting.util import (
    load_binary_pickle,  # 현재 예제에선 직접 호출 X (참고용)
    write_simple_obj,    # 버텍스+페이스 → .obj 로 저장
    safe_mkdir,          # 폴더 생성 (존재하면 PASS)
    get_unit_factor,     # 단위(m/cm/mm) 변환 factor
)
from typing import Optional

# -----------------------------------------------------------------------------
#                                핵심 함수
# -----------------------------------------------------------------------------

def fit_lmk3d(
    lmk_3d: np.ndarray,          # (N,3) 형태의 입력 랜드마크 (meters 단위 권장)
    model,                       # FLAME chumpy 모델 객체
    lmk_face_idx: np.ndarray,    # 각 랜드마크가 속한 메쉬 삼각형(face) 인덱스 (shape: N)
    lmk_b_coords: np.ndarray,    # 각 랜드마크의 barycentric 좌표 (shape: N,3)
    weights: dict,               # 손실 함수(목적 함수)별 가중치
    shape_num: int = 300,        # 사용하고 싶은 identity(shape) basis 개수 (0~299)
    expr_num: int = 100,         # 사용하고 싶은 expression basis 개수 (300~399)
    opt_options: Optional[dict] = None,
):
    """FLAME 모델을 3D 랜드마크에 맞게 피팅(fitting)한다.

    ----------------------------------------------------------------------------------
    입력 변수 설명
    ----------------------------------------------------------------------------------
    lmk_3d : (N,3)
        • 3차원 공간상의 랜드마크 좌표.
        • 예) Mediapipe FaceMesh → 좌표 스케일 보정 후 저장한 .npy 파일.
    model  : chumpy object
        • load_model() 로 로드한 FLAME.
        • 내부에 pose(관절 회전), betas(모프 계수), trans(평행이동) 등이 포함.
    lmk_face_idx, lmk_b_coords :
        • "어떤 랜드마크가 FLAME 메쉬의 어떤 삼각형 안 어디쯤에 있는가"를 미리 계산한 테이블.
        • FLAME 공식 배포본에서 제공.
    weights : dict
        • 'lmk', 'shape', 'expr', 'pose' 네 항목의 스칼라값.
        • lmk 오차를 줄이면서도, 과도한 왜곡(overshoot)을 막기 위해 regularizer 가중치 부여.
    shape_num / expr_num :
        • basis 전체(300/100)를 다 쓰면 오버피팅 위험 → sparse 랜드마크엔 보통 100/50 정도만 사용.
    opt_options :
        • scipy.optimize 기반 Dogleg solver 세부 옵션.
        • None이면 기본값 세팅.

    ----------------------------------------------------------------------------------
    출력 값 설명
    ----------------------------------------------------------------------------------
    (mesh_v, mesh_f, parms)
        mesh_v : (V,3) array  → 피팅된 최종 버텍스 위치
        mesh_f : (F,3) array  → 삼각형 face 인덱스 (모델마다 동일)
        parms  : dict         → 'pose', 'betas', 'trans' 파라미터 벡터 복사본
    """

    # ------------------------------------------------------------------
    # 1) 최적화에 사용할 파라미터 인덱스 계산
    # ------------------------------------------------------------------
    # pose 벡터 구조: [0:3]=global rot, [3:6]=neck, [6:9]=jaw, 그 뒤로 eye rot ...
    pose_idx = np.union1d(np.arange(3), np.arange(6, 9))  # 전역 회전 + 턱 회전만 자유 변수로.

    # betas 벡터 구조: [0:300)=shape, [300:400)=expression
    shape_idx = np.arange(0, min(300, shape_num))
    expr_idx = np.arange(300, 300 + min(100, expr_num))
    used_idx = np.union1d(shape_idx, expr_idx)

    # ------------------------------------------------------------------
    # 2) 모델 파라미터 초기화
    # ------------------------------------------------------------------
    model.betas[:] = 0.0  # identity & expression 모두 평균 얼굴로 시작
    model.pose[:] = 0.0   # 모든 관절 회전 0으로 시작 (즉, 정면)

    # 최적화 대상 리스트 (Dogleg solver에게 전달)
    free_variables = [model.trans, model.pose[pose_idx], model.betas[used_idx]]

    # ------------------------------------------------------------------
    # 3) 목적 함수(loss) 정의
    # ------------------------------------------------------------------
    # 3‑1) landmark 재투영 오차 (가장 핵심!)
    lmk_err = landmark_error_3d(
        mesh_verts=model,          # 현재 FLAME 버텍스 (chumpy 연산 가능)
        mesh_faces=model.f,        # 페이스 인덱스
        lmk_3d=lmk_3d,            # 타겟 랜드마크
        lmk_face_idx=lmk_face_idx,
        lmk_b_coords=lmk_b_coords,
        weight=weights["lmk"],
    )

    # 3‑2) Regularizer: 과도한 왜곡 방지용 가중치
    shape_err = weights["shape"] * model.betas[shape_idx]
    expr_err = weights["expr"] * model.betas[expr_idx]
    # pose[3:] : 글로벌 회전 제외한 나머지 회전에 규제 적용
    pose_err = weights["pose"] * model.pose[3:]

    # 목적 함수 dict (다중 항목 최적화)
    objectives = {"lmk": lmk_err, "shape": shape_err, "expr": expr_err, "pose": pose_err}

    # ------------------------------------------------------------------
    # 4) 최적화 옵션 세팅 (Dogleg + Conjugate Gradient)
    # ------------------------------------------------------------------
    if opt_options is None:
        import scipy.sparse as sp

        opt_options = {
            "disp": 1,              # 진행 상황 출력 여부
            "delta_0": 0.1,         # 신뢰 영역 초기 반경
            "e_3": 1e-4,            # 종료 기준 (gradient norm)
            "maxiter": 2000,
            "sparse_solver": lambda A, x: sp.linalg.cg(A, x, maxiter=2000)[0],
        }

    # 빈 콜백 (원하면 중간 과정을 시각화할 때 수정)
    def on_step(_):
        pass

    # ------------------------------------------------------------------
    # 5) Rigid Fitting  ── 전역 위치/방향 맞추기
    # ------------------------------------------------------------------
    import time

    print("\n[Step 1] Rigid fitting …")
    t0 = time.time()
    ch.minimize(
        fun=lmk_err,
        x0=[model.trans, model.pose[0:3]],  # 최적화 변수 제한
        method="dogleg",
        callback=on_step,
        options=opt_options,
    )
    print(f"[Step 1] Done in {time.time() - t0:.2f} sec\n")

    # ------------------------------------------------------------------
    # 6) Non‑Rigid Fitting  ── 세부 shape / expression / jaw 조정
    # ------------------------------------------------------------------
    print("[Step 2] Non‑rigid fitting …")
    t0 = time.time()
    ch.minimize(
        fun=objectives,
        x0=free_variables,
        method="dogleg",
        callback=on_step,
        options=opt_options,
    )
    print(f"[Step 2] Done in {time.time() - t0:.2f} sec\n")

    # ------------------------------------------------------------------
    # 7) 결과 반환
    # ------------------------------------------------------------------
    parms = {
        "trans": model.trans.r.copy(),
        "pose": model.pose.r.copy(),
        "betas": model.betas.r.copy(),
    }
    return model.r.copy(), model.f.copy(), parms

# -----------------------------------------------------------------------------
#                                진입점 (CLI)
# -----------------------------------------------------------------------------

def run_fitting():
    """샘플 .npy 랜드마크를 불러와 FLAME 피팅을 실행한다."""

    # 1) 랜드마크 로드 + 단위 보정 ------------------------------------------------
    lmk_path = "./data/scan_lmks.npy"  # 준비된 (N,3) npy
    lmk_path = "./data/test_lmks.npy"  # 준비된 (N,3) npy
    # lmk_path = "./data/jinju_lmks.npy"  # 준비된 (N,3) npy
    # lmk_path = "./data/madong_lmks.npy"  # 준비된 (N,3) npy
    unit = "m"                            # 현 파일의 단위를 meter로 가정
    lmk_raw = np.load(lmk_path)
    scale_factor = get_unit_factor("m") / get_unit_factor(unit)
    lmk_3d = scale_factor * lmk_raw
    print("loaded 3d landmark from:", lmk_path, "→ shape:", lmk_3d.shape)

    # 2) FLAME 모델 로드 --------------------------------------------------------
    model_path = "./models/flame2023.pkl"  # gender 알면 male/female 로 교체 가능
    model = load_model(model_path)
    print("loaded model from:", model_path)

    # 3) 랜드마크 임베딩 로드 -----------------------------------------------------
    emb_path = "./models/flame_static_embedding.pkl"
    lmk_face_idx, lmk_b_coords = load_embedding(emb_path)
    print("loaded landmark embedding (face_idx, b_coords)")

    # 4) 출력 디렉토리 준비 -------------------------------------------------------
    output_dir = "./output"
    safe_mkdir(output_dir)

    # 5) 손실 가중치 & basis 개수 설정 -------------------------------------------
    weights = {"lmk": 1.0, "shape": 1e-43, "expr": 1e-4, "pose": 1e-4}
    shape_num, expr_num = 100, 50

    # 6) 피팅 실행 --------------------------------------------------------------
    verts, faces, parms = fit_lmk3d(
        lmk_3d=lmk_3d,
        model=model,
        lmk_face_idx=lmk_face_idx,
        lmk_b_coords=lmk_b_coords,
        weights=weights,
        shape_num=shape_num,
        expr_num=expr_num,
    )

    # 7) 결과 저장 --------------------------------------------------------------
    obj_path = join(output_dir, "fit_lmk3d_result.obj")
    write_simple_obj(verts, faces, obj_path, verbose=False)
    print("Saved fitted mesh →", obj_path)

    # 파라미터도 추후 시각화/재사용을 위해 같이 저장할 수 있음.

# -----------------------------------------------------------------------------
#                                  실행 스크립트
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    run_fitting()
