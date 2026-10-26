from typing import Callable

def nelder_mead(penalized_ssr_score : Callable[[float, float], float], lambda_1 : float, lambda_2 : float, initial_shift : float, x_rtol : float, f_rtol : float, max_iteration : int) -> tuple[float, float]:
    """
    2-D Nelder-Mead refinement of (lambda_1, lambda_2) around a starting
    point, minimizing penalized_ssr_score.

    Standard reflect/expand/contract/shrink simplex updates. The initial
    simplex shifts each lambda by initial_shift. penalized_ssr_score is
    expected to return +inf outside the admissible region (lambda_1 > 0,
    lambda_2 > lambda_1 + 1, lambda_2 <= LAMBDA_MAX), so the simplex avoids it.

    Stops when the relative spread of the vertices around the best one is
    at most x_rtol and the SSR spread is at most f_rtol * best SSR, and
    returns the best vertex.
    Raises RuntimeError if not converged after max_iteration iterations.
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

        x_spread=max(abs(second_best[0]-best[0])/best[0], abs(second_best[1]-best[1])/best[1],
                     abs(worst[0]-best[0])/best[0], abs(worst[1]-best[1])/best[1])
        ssr_spread=ssr_worst-ssr_best

        if x_spread<=x_rtol and ssr_spread<=f_rtol*ssr_best:
            return best

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

    raise RuntimeError("Nelder-Mead did not converge: tolerance not reached after max_iteration iterations.")