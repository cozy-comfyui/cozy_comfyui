""" Maths Easing Function Support """

def seriesLinear(start=0.0, step=1.0, count=10, loop=None, pingpong=False) -> list[float]:
    result = []

    # Calculate how many steps in each loop
    count = max(1, count)
    loop_size = loop or count

    for i in range(count):
        # Calculate which step within the loop we're at
        if loop is not None:
            loop_position = i % loop_size
        else:
            loop_position = i

        # Calculate the raw value based on start and step
        if pingpong:
            # For ping-pong, we need to determine if we're in a forward or backward cycle
            if loop is not None:
                # With loop, we ping-pong within each loop
                cycle_length = loop_size * 2 - 2  # Total steps in a full ping-pong cycle
                cycle_position = i % cycle_length if cycle_length > 0 else 0

                if cycle_position < loop_size:
                    # Forward part of the cycle
                    current = start + cycle_position * step
                else:
                    # Backward part of the cycle
                    loop_end = start + (loop_size - 1) * step
                    steps_from_end = cycle_position - loop_size + 1
                    current = loop_end - steps_from_end * step
            else:
                # Without loop, ping-pong based on count
                cycle_length = count * 2 - 2  # Total steps in a full ping-pong cycle
                cycle_position = i % cycle_length if cycle_length > 0 else 0

                if cycle_position < count:
                    # Forward part of the cycle
                    current = start + cycle_position * step
                else:
                    # Backward part of the cycle
                    count_end = start + (count - 1) * step
                    steps_from_end = cycle_position - count + 1
                    current = count_end - steps_from_end * step
        else:
            # Normal progression with looping
            current = start + loop_position * step
        result.append(current)

    return result
