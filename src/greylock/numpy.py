from numpy import divide, abs


def get_community_ratio(numerator, denominator):
    return divide(
        numerator,
        denominator,
        out=zeros(denominator.shape),
        where=denominator != 0,
    )

