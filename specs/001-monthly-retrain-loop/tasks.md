# Tasks: Monthly retraining on real data, fair promotion and performance monitoring

## Phase 1: Foundation
- [x] T001 `nyc_tip/promotion.py`: fingerprint, paired bootstrap of AUC difference, decision
- [x] T002 [P] Tests for the rule and the bootstrap

## Phase 2: User Story 1 (P1)
- [x] T003 `train_v2.run` scores the new model and the registry champion on a holdout and promotes by the rule
- [x] T004 [P] Registry tests updated (first promoted, equal not promoted, decision recorded)

## Phase 3: User Story 2 (P2)
- [x] T005 `training/retrain.py`: newest month, train on L-1, compare on L, rewrite the bundle, history
- [x] T006 [P] Loop tests on synthetic months
- [x] T007 `retrain.yml` monthly on real data; remove the synthetic weekly retrain

## Phase 4: User Story 3 (P3)
- [x] T008 Performance history and floor alert in the loop and the workflow
- [x] T009 Dashboard shows the performance history and the last decision

## Phase 5: User Story 4 (P4)
- [x] T010 API key on prediction endpoints; tests

## Phase 6: Polish
- [x] T011 Run the loop on real months; README, ADRs 0006-0007
