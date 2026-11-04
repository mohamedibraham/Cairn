

PRICING_USD_PER_1M: dict[tuple[str, str], tuple[float, float]] = {
    
    ("gemini", "gemini-2.5-flash"):          (0.30,  2.50),
    ("gemini", "gemini-2.5-flash-preview"):  (0.30,  2.50),
    ("gemini", "gemini-2.5-pro"):            (1.25, 10.00),
    ("gemini", "gemini-2.0-flash"):          (0.10,  0.40),

    
    ("cohere", "command-r-plus-08-2024"):    (2.50, 10.00),
    ("cohere", "command-r-08-2024"):         (0.15,  0.60),
    ("cohere", "command-r7b-12-2024"):       (0.0375, 0.15),
    ("cohere", "command-a-03-2025"):         (2.50, 10.00),
}


def price_for(provider: str, model: str) -> tuple[float, float] | None:
    
    return PRICING_USD_PER_1M.get((provider.lower(), model.lower()))


def estimate_cost(provider: str, model: str,
                  input_tokens: int, output_tokens: int) -> float | None:
   
    prices = price_for(provider, model)
    if prices is None:
        return None
    in_price, out_price = prices
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000
