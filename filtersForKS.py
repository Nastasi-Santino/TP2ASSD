def simple_average(input_sample, prev_output):
    """
    Implements: y(n) = 0.5 * (x(n) + 0.5 * y(n-1))
    
    Args:
        input_sample: Current input to the filter x(n)
        prev_output: Previous output y(n-1) from z^-1 delay
    
    Returns:
        Filtered output sample y(n)
    """
    output = 0.5 * (input_sample + 0.5 * prev_output)
    return output

def recursive_LP_filter(input_sample, prev_output, gain, alpha):
    """
    Implements: y(n) = gain * [(1 - alpha) * x(n) + alpha * y(n-1)]

    Args:
        input_sample: Current input to the filter x(n)
        prev_output: Previous output y(n-1) from z^-1 delay
        gain: Overall gain factor for the filter
        alpha: Feedback coefficient (0 < alpha < 1)

    Returns:
        Filtered output sample y(n)
    """
    output = gain*(1-alpha)*input_sample + alpha*prev_output
    return output

def recursive_AP_filter(input_sample,prev_input, prev_output, alpha):
    """
    Implements: w(n) = v(n) - 0.5 * v(n-1)

    Args:
        input_sample: Current input to the allpass filter v(n)
        prev_input: Previous input v(n-1) from z^-1 delay
        prev_output: Previous output v(n-1) from z^-1 delay
        alpha: Allpass filter coefficient (0 < alpha < 1)

    Returns:
        Allpass filtered output sample w(n)
    """
    output = -alpha * input_sample + prev_input + alpha * prev_output 
    return output