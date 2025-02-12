---
title: SQL Window Function
date: 2025-02-12 12:35:16 +09:00
categories: ['SQL']
tags: ['MySQL', 'Window Function']
---

# SQL Window Function

SQL에서 데이터를 다룰 때, 단순한 그룹화와 집계 함수만으로 원하는 결과를 얻기 어려운 경우가 많습니다. 
특정 범위 내에서 순위를 매기거나 누적 합을 계산하는 등 보다 정교한 분석이 필요할 때 윈도우 함수(Window Functions)를 사용합니다. 
윈도우 함수는 기존의 집계 함수와 달리, GROUP BY 없이도 개별 행을 유지한 상태에서 추가적인 분석을 수행할 수 있습니다.

코딩 테스트에서도 SQL 관련 문제가 출제되며, 윈도우 함수는 데이터를 보다 효과적으로 다룰 수 있는 핵심 기능 중 하나입니다.
본 글에서는 윈도우 함수의 기본 개념과 자주 사용되는 함수들을 정리합니다.

## 윈도우 함수 기본 구조
```sql
SELECT WINDOW_FUNCTION(ARGUMENTS)
  OVER (PARTITION BY COLUMN ORDER BY WINDOWING)
FROM TableName;
```
#### 구조 설명
| 요소 | 설명 |
| --- | --- |
| ARGUMENTS(인수) | 윈도우 함수에 따라 0개 이상의 인수를 설정한다. |
| PARTITION BY | 전체 집합을 특정 기준에 따라 소그룹으로 나눈다. |
| ORDER BY | 특정 컬럼을 기준으로 정렬한다. |
| WINDOWING | 행 기준 범위를 지정한다. `ROWS`는 물리적 행 수를, `RANGE`는 논리적 값의 범위를 설정한다. |

## WINDOWING 옵션
윈도우 함수에서 `OVER` 절의 `ORDER BY`와 함께 사용할 수 있는 `WINDOWING`은 윈도우 크기를 지정합니다.

| 옵션 | 설명 |
| --- | --- |
| ROWS | 윈도우 크기를 물리적 행 단위로 지정한다. |
| RANGE | 논리적 값 기준으로 윈도우 크기를 지정한다. |
| BETWEEN~AND | 윈도우의 시작과 끝 위치를 지정한다. |
| UNBOUNDED PRECEDING | 윈도우의 시작 위치가 첫 번째 행임을 의미한다. |
| UNBOUNDED FOLLOWING | 윈도우의 마지막 위치가 마지막 행임을 의미한다. |
| CURRENT ROW | 현재 행을 기준으로 윈도우를 설정한다. |

## 윈도우 함수
윈도우 함수는 크게 네 가지 유형으로 구분할 수 있습니다.

### 1. 순위 함수 (Rank Functions)
- `RANK()`: 동일한 값에 동일한 순위를 부여하되, 다음 순위를 건너뛴다.
- `DENSE_RANK()`: 동일한 값에 동일한 순위를 부여하지만, 순위가 연속된다.
- `ROW_NUMBER()`: 중복 없이 순위를 부여한다.

### 2. 집계 함수 (Aggregate Functions)
- `SUM()`: 누적 합을 구한다.
- `AVG()`: 평균을 계산한다.
- `COUNT()`: 행의 개수를 센다.
- `MAX(), MIN()`: 최댓값 및 최솟값을 구한다.

### 3. 순서 함수 (Order Functions)
- `FIRST_VALUE()`: 윈도우 내 첫 번째 값을 반환한다.
- `LAST_VALUE()`: 윈도우 내 마지막 값을 반환한다.
- `LAG()`: 이전 행의 값을 가져온다.
- `LEAD()`: 다음 행의 값을 가져온다.

### 4. 비율 함수 (Ratio Functions)
- `CUME_DIST()`: 누적 분포를 반환한다.
- `PERCENT_RANK()`: 백분율 순위를 계산한다.
- `NTILE(N)`: 데이터를 N개 그룹으로 나눈다.
- `RATIO_TO_REPORT()`: 값이 전체 합에서 차지하는 비율을 계산한다.


> 윈도우 함수는 SQL에서 데이터를 유연하고 효율적으로 다룰 수 있도록 도와주는 중요한 기능입니다. 
> 개별 행을 유지하면서도 순위 계산, 누적 합, 비율 분석 등을 수행할 수 있어 실무뿐만 아니라 코딩 테스트에서도 유용하게 활용됩니다. 
> 본 글이 도움이 되었기를 바랍니다! ✌️✌️

