import torch
import torch.nn.functional as F

class distill_loss:
    def __init__(self,student_model,teacher_model,alpha,T):
        self.student = student_model
        self.teacher = teacher_model
        self.alpha = alpha
        self.T = T

    def forward(self, input,pad_tensor,hard_label):
        pad_mask = torch.einsum("bit,bjt->bij", pad_tensor.float().unsqueeze(-1),pad_tensor.float().unsqueeze(-1))
        pad_mask = (1 - pad_mask) * -1e8
        stu_logits = self.student(input,pad_mask) / self.T
        tcr_logits = self.teacher(input,pad_mask) / self.T
        shift_stu_logits = stu_logits[..., :-1, :].contiguous()
        shift_tcr_logits = tcr_logits[..., :-1, :].contiguous()
        hard_label = hard_label*pad_tensor
        hard_label[hard_label == 0] = -100
        shift_labels = hard_label[..., 1:].contiguous()

        kl_dim = F.kl_div(F.log_softmax(shift_stu_logits.view(-1,shift_stu_logits.size(-1)),dim=-1),
                             F.log_softmax(shift_tcr_logits.view(-1,shift_tcr_logits.size(-1)),dim=-1),
                             reduction='none',
                             log_target=True).sum(dim=-1)
        masked_kl = kl_dim * pad_tensor.unsqueeze(-1)
        soft_loss = masked_kl.sum() / pad_tensor.sum()

        hard_loss = F.cross_entropy(shift_stu_logits.view(-1, shift_stu_logits.size(-1)),
                                    shift_labels.view(-1),
                                    ignore_index=-100)

        total_loss = hard_loss * (1 - self.alpha) + soft_loss * self.alpha
        return total_loss


    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

