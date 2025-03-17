def wrap_text(text, max_length=80, separator="\n"):
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 > max_length:
            lines.append(current_line.strip())
            current_line = word
        else:
            current_line += " " + word

    if current_line:
        lines.append(current_line.strip())
    
    return separator.join(lines)
