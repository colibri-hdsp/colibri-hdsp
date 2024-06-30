import torch
import time
from tqdm import tqdm


class Training:
    """
    Class for training a neural network model.
    """

    def __init__(
        self,
        model,
        train_loader,
        optimizer = None,
        loss_func = {"MSE": torch.nn.MSELoss()},
        losses_weights = [1.0],
        metrics = {},
        regularizers = {},
        regularizers_optics_mo = {},
        regularization_optics_weights_mo = {},
        regularizers_optics_ce = {},
        regularization_optics_weights_ce = {},
        regularization_weights = {},
        schedulers=[],
        callbacks=[],
        device="cpu",
    ):
        """
        Args:
            model (torch.nn.Module): Neural network model.
            train_loader (torch.utils.data.DataLoader): Training data loader.
            optimizer (torch.optim): Optimizer.
            loss_func (dict): Dictionary of loss functions, format: {"name_loss":function}.
            losses_weights (list): List of weights for each loss function.
            metrics (dict): Dictionary of metrics, format: {"name_metric":function}.
            regularizers (dict): Dictionary of regularizers, format: {"name_regularizer":function}.
            regularization_weights (list): List of weights for each regularizer.
            schedulers (list): List of learning rate schedulers.
            callbacks (list): List of callbacks.
            regularizers_optics(dict): Dictionary of regularizers for optics, format: {"name_regularizer":function}.
            device (str): Device to use for training.
        """
        self.model = model
        self.train_loader = train_loader
        self.loss_func = loss_func
        self.losses_weights = losses_weights

        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.optimizer = optimizer
        self.metrics = metrics
        self.regularizers = regularizers
        self.regularizers_optics_mo = regularizers_optics_mo
        self.regularization_weights = regularization_weights
        self.regularization_optics_weights_mo = regularization_optics_weights_mo
        self.regularizers_optics_ce = regularizers_optics_ce
        self.regularization_optics_weights_ce = regularization_optics_weights_ce
        self.schedulers = schedulers
        self.callbacks = callbacks
        self.device = device

    def train_one_epoch(self, freq=1, steps_per_epoch=None, tq=None):
        """
        Train model for one epoch.

        Args:
            freq (int): Frequency for printing training progress.
            steps_per_epoch (int): Number of steps per epoch.
        Returns:
            dict: Dictionary of loss values.
        """

        ## Compute time for each batch
        start_time = time.time()
        for i, data in enumerate(self.train_loader):
            tq.update(1)
            # Every data instance is an input + outputs pair

            inputs = data['input']
            outputs_gt = inputs.clone()
            inputs = inputs.to(self.device)
            outputs_gt = outputs_gt.to(self.device)

            # Zero your gradients for every batch!
            self.optimizer.zero_grad(set_to_none=True)

            # Make inference
            outputs_pred = self.model(inputs)

            final_loss = 0.0
            loss_values = (
                {}
            )  # loss_values = { key: 0.0 for key in self.loss_func.keys()}
            for idx, key in enumerate(self.loss_func.keys()):

                res = (
                    self.loss_func[key](outputs_pred, outputs_gt)
                    * self.losses_weights[idx]
                )
                loss_values[key] = res
                final_loss += loss_values[key]
            txt_losses = ""
            txt_reg_ce = ""
            txt_reg_mo = ""
            txt_reg_tot = None
            total_reg = {}
            if self.regularizers is not None:
                tmp, reg_decoders = self.reg_decoder()
                final_loss += tmp
                txt_reg_decoders = "".join(
                    [f"{key}: {reg_decoders[key]:.2E}, " for key in reg_decoders.keys()]
                )
                if txt_reg_tot is None:
                    txt_reg_tot = txt_reg_decoders
                else:
                    txt_reg_tot = f"{txt_reg_tot}, {txt_reg_decoders}"
                total_reg.update(reg_decoders)
            if self.regularizers_optics_ce is not None:
                tmp, reg_ce = self.reg_optics_ce(inputs)
                final_loss += tmp
                txt_reg_ce = "".join(
                    [f"{key}: {reg_ce[key]:.2E}, " for key in reg_ce.keys()]
                )
                total_reg.update(reg_ce)
                if txt_reg_tot is None:
                    txt_reg_tot = txt_reg_ce
                else:
                    txt_reg_tot = f"{txt_reg_tot}, {txt_reg_ce}"
            if self.regularizers_optics_mo is not None:
                tmp, reg_mo = self.reg_optics_mo(inputs)
                final_loss += tmp
                txt_reg_mo = "".join(
                    [f"{key}: {reg_mo[key]:.2E}, " for key in reg_mo.keys()]
                )
                total_reg.update(reg_mo)
                if txt_reg_tot is None:
                    txt_reg_tot = txt_reg_mo
                else:
                    txt_reg_tot = f"{txt_reg_tot}, {txt_reg_mo}"

            final_loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Keep track of loss

            with torch.no_grad():
                final_metric = 0.0
                metric_values = {}
                for idx, key in enumerate(self.metrics.keys()):
                    res = self.metrics[key](outputs_pred, outputs_gt)
                    metric_values[key] = res
                    final_metric += metric_values[key]
            # Elapsed time

            elapsed_time = time.time() - start_time
            start_time = time.time()
            t = elapsed_time / freq

            txt_losses = "".join(
                [f"{key}: {loss_values[key]:.2E}, " for key in loss_values.keys()]
            )

            # print(f'  batch {i + 1}/{len(self.train_loader)}, {txt_losses}, {txt_reg_tot}, time per batch: {t:.1f} [s]')
            tq.set_postfix(s=f"{txt_losses}, {txt_reg_tot}")

            if steps_per_epoch != None and i >= steps_per_epoch:
                return loss_values, total_reg

        return loss_values, total_reg

    def reg_decoder(self, verbose=False):
        """
        Weight regularization for one epoch.
        """
        running_reg = 0.0
        reg_values = {}
        for idx, key in enumerate(self.regularizers.keys()):
            for p in self.model.decoder.parameters():
                if p.requires_grad:
                    reg = self.regularizers[key](p) * self.regularization_weights[idx]
                    reg_values[key] = reg
                    running_reg += reg
        if verbose and len(reg_values) > 0:
            print(f"  regularization loss: {running_reg:.5E}")
        return running_reg, reg_values

    def reg_optics_ce(self, x=None, verbose=False):
        reg_values = {}
        running_reg = 0.0

        for idx, key in enumerate(self.regularizers_optics_ce.keys()):

            reg = (
                self.model.optical_layer.weights_reg(self.regularizers_optics_ce[key])
                * self.regularization_optics_weights_ce[idx]
            )
            reg_values[key] = reg
            running_reg += reg
        if verbose and len(reg_values) > 0:
            print(f"  optics  regularization on ce loss: {running_reg:.5E}")
        return running_reg, reg_values

    def reg_optics_mo(self, x=None, verbose=False):
        reg_values = {}
        running_reg = 0.0

        for idx, key in enumerate(self.regularizers_optics_mo.keys()):

            reg = (
                self.model.optical_layer.output_reg(
                    self.regularizers_optics_mo[key], x
                )
                * self.regularization_optics_weights_mo[idx]
            )
            reg_values[key] = reg
            running_reg += reg
        if verbose and len(reg_values) > 0:
            print(f"  optics  regularization on middle output loss: {running_reg:.5E}")
        return running_reg, reg_values

    def fit(self, n_epochs, verbose_reg=True, freq=1, steps_per_epoch=None):
        """
        Train model

        Args:
            n_epochs (int): Number of epochs.
            freq (int): Frequency for printing training progress.
            steps_per_epoch (int): Number of steps per epoch.
        Returns:
            dict: Dictionary of loss values.
        """
        start_time = time.time()
        for epoch in range(n_epochs):

            with tqdm(
                total=len(self.train_loader), dynamic_ncols=True, colour="blue"
            ) as tq:
                tq.set_description(f"Train :: Epoch: {epoch + 1}/{n_epochs}")

                # print('Epoch {}/{}'.format(epoch + 1, n_epochs))

                self.model.train(True)
                if self.loss_func:
                    results_fidelities, total_reg = self.train_one_epoch(
                        freq=freq, steps_per_epoch=steps_per_epoch, tq=tq
                    )
                else:
                    results_fidelities = {}

                results_losses = {**results_fidelities, **total_reg}
                self.model.train(False)

                for s in self.schedulers:
                    s.step()

                for c in self.callbacks:
                    c.step(self.model, results_losses, epoch)

                elapsed_time = time.time() - start_time
                start_time = time.time()
                # print('  time per epoch: {:.1f} [s]'.format(elapsed_time))

        return results_losses
