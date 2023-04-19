import torch

# TODO make this perform early stopping based on test vs train accuracy

class CircuitBreaker:
    def __init__(self, model, device: str):
        self.model = model
        self.cont = True
        self.loss_total = 0
        self.total_ct = 0
        self.device = device
        self.crit = torch.nn.CrossEntropyLoss()
        self.delta: list[float] = [100000000.0]

    def ok(self) -> bool:
        return self.cont

    def update(self, loader):
        with torch.no_grad():
            cur_loss = 0
            cur_tot = 0
            for (images, labels) in loader:
                images = images.float().to(self.device)
                labels = labels.to(self.device)

                loss = self.crit(self.model(images), labels).item()
                cur_loss += loss
                cur_tot += 1

            cur = cur_loss/cur_tot
            tot = self.loss_total/self.total_ct

            if self.delta[-1] < cur-tot:
                self.cont = False
            self.delta.append(cur-tot)

            self.loss_total += cur_loss
            self.total_ct += cur_tot
