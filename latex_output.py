import numbers


def _format_cell(value, decimals):
  # tuple metrics (e.g. (mean, std)) -> "value \pm value" latex format
  if isinstance(value, tuple):
    if len(value) == 2:
      mean, std = value
      return f"${float(mean):.{decimals}f} \\pm {float(std):.{decimals}f}$"
    return "(" + ", ".join(f"{float(v):.{decimals}f}" for v in value) + ")"

  # plain numbers (including np.float64/np.int64) -> normal-looking decimal format
  if isinstance(value, numbers.Number):
    return f"{float(value):.{decimals}f}"

  return value


def print_latex_table(df, caption, label, decimals=4):
  formatted_df = df.map(lambda v: _format_cell(v, decimals))

  print(f"% {caption}")
  print(
      formatted_df.to_latex(
          index=False,
          caption=caption,
          label=label,
      )
  )
