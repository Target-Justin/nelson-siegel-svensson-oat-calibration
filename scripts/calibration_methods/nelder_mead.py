from typing import Callable

def nelder_mead(penalized_ssr_score : Callable[[float, float], float], lambda_1 : float, lambda_2 : float, initial_shift : float, max_iteration : int) -> tuple[float, float]:
    """
    2-D Nelder-Mead refinement of (lambda_1, lambda_2) around a starting
    point, minimizing penalized_ssr_score.

    Standard reflect/expand/contract/shrink simplex updates;
    penalized_ssr_score is expected to return +inf outside the admissible
    region (the NSS identifiability constraint lambda_2 > lambda_1 + 1),
    so the simplex avoids it. Returns the best (lambda_1, lambda_2) found
    after max_iteration steps.
    """
    
    u=(lambda_1, lambda_2)
    v=(lambda_1+initial_shift, lambda_2)
    w=(lambda_1, lambda_2+initial_shift)

    ssr_u=penalized_ssr_score(u[0], u[1])
    ssr_v=penalized_ssr_score(v[0], v[1])
    ssr_w=penalized_ssr_score(w[0], w[1])

    list_sorted=[(u,ssr_u),(v,ssr_v),(w,ssr_w)]
    list_sorted.sort(key=lambda x: x[1])

    for _ in range(max_iteration):

        best, ssr_best=list_sorted[0]
        second_best, ssr_second_best=list_sorted[1]
        worst, ssr_worst=list_sorted[2]

        centroid=((best[0]+second_best[0])/2, (best[1]+second_best[1])/2)
        reflected=(2*centroid[0]-worst[0], 2*centroid[1]-worst[1])
        ssr_reflected=penalized_ssr_score(reflected[0], reflected[1])

        if ssr_reflected < ssr_best:

            expanded=(2*reflected[0]-centroid[0], 2*reflected[1]-centroid[1])
            ssr_expanded=penalized_ssr_score(expanded[0], expanded[1])

            if ssr_expanded < ssr_reflected:

                list_sorted[2]=list_sorted[1]
                list_sorted[1]=list_sorted[0]
                list_sorted[0]=(expanded, ssr_expanded)

                
            else:

                list_sorted[2]=list_sorted[1]
                list_sorted[1]=list_sorted[0]
                list_sorted[0]=(reflected, ssr_reflected)

        elif ssr_reflected < ssr_second_best:

            list_sorted[2]=list_sorted[1]
            list_sorted[1]=(reflected, ssr_reflected)

        else:

            contracted=((centroid[0]+worst[0])/2, (centroid[1]+worst[1])/2)
            ssr_contracted=penalized_ssr_score(contracted[0], contracted[1])

            if ssr_contracted < ssr_worst:

                list_sorted[2]=(contracted, ssr_contracted)

            else:

                new_second_best=((best[0]+second_best[0])/2, (best[1]+second_best[1])/2)
                new_worst=((best[0]+worst[0])/2, (best[1]+worst[1])/2)
                ssr_new_second_best=penalized_ssr_score(new_second_best[0], new_second_best[1])
                ssr_new_worst = penalized_ssr_score(new_worst[0], new_worst[1])

                list_sorted[2] = (new_worst, ssr_new_worst)
                list_sorted[1] = (new_second_best, ssr_new_second_best)

            list_sorted.sort(key=lambda x: x[1])

    return list_sorted[0][0]