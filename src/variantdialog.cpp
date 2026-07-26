#include "variantdialog.h"

#include <QDialogButtonBox>
#include <QFileInfo>
#include <QLabel>
#include <QListWidget>
#include <QPixmap>
#include <QPushButton>
#include <QVBoxLayout>

static constexpr int THUMB_SIZE = 220;

VariantDialog::VariantDialog(const QStringList &imagePaths, QWidget *parent)
    : QDialog(parent), paths(imagePaths)
{
    setWindowTitle(tr("Choose a Variation"));
    setWindowFlag(Qt::WindowMaximizeButtonHint);

    auto *heading = new QLabel(tr("%1 variations generated — pick the one to keep.")
                                       .arg(imagePaths.size()),
                               this);

    list = new QListWidget(this);
    list->setViewMode(QListView::IconMode);
    list->setIconSize(QSize(THUMB_SIZE, THUMB_SIZE));
    list->setGridSize(QSize(THUMB_SIZE + 24, THUMB_SIZE + 44));
    list->setResizeMode(QListView::Adjust);
    list->setMovement(QListView::Static);
    list->setSpacing(6);
    list->setSelectionMode(QAbstractItemView::SingleSelection);
    list->setUniformItemSizes(true);

    for (int i = 0; i < paths.size(); ++i) {
        QPixmap thumb(paths.at(i));
        if (thumb.isNull())
            continue;
        auto *item = new QListWidgetItem(
                QIcon(thumb.scaled(THUMB_SIZE, THUMB_SIZE, Qt::KeepAspectRatio,
                                   Qt::SmoothTransformation)),
                tr("Variation %1").arg(i + 1));
        item->setData(Qt::UserRole, paths.at(i));
        item->setTextAlignment(Qt::AlignHCenter);
        list->addItem(item);
    }

    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    okButton = buttons->button(QDialogButtonBox::Ok);
    okButton->setText(tr("Use This One"));
    okButton->setEnabled(false);

    auto *layout = new QVBoxLayout(this);
    layout->addWidget(heading);
    layout->addWidget(list, 1);
    layout->addWidget(buttons);

    connect(list, &QListWidget::itemSelectionChanged, this,
            [this]() { okButton->setEnabled(!list->selectedItems().isEmpty()); });
    // Double-clicking a thumbnail is the fast path: pick it and close.
    connect(list, &QListWidget::itemDoubleClicked, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

    if (list->count() > 0)
        list->setCurrentRow(0);

    // Fit up to three thumbnails per row without forcing a huge window.
    const int columns = qMin(3, qMax(1, list->count()));
    resize(qMin(1100, columns * (THUMB_SIZE + 40) + 60), 620);
}

QString VariantDialog::selectedPath() const
{
    const auto selection = list->selectedItems();
    if (selection.isEmpty())
        return QString();
    return selection.first()->data(Qt::UserRole).toString();
}
