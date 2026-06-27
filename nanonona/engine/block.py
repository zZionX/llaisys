class Block:
    def __init__(self, block_id: int = -1):
        self.block_id = block_id
        self.token_ids = []
        self.hash = -1
        self.ref_count = 0