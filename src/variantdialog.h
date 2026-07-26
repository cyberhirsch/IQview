#ifndef VARIANTDIALOG_H
#define VARIANTDIALOG_H

#include <QDialog>
#include <QStringList>

class QListWidget;
class QPushButton;

// VariantDialog — shown after a batch Creative Fill so the user can pick which
// generated variation to keep. Displays each result as a thumbnail; the chosen
// one is applied and the rest are discarded.
class VariantDialog : public QDialog
{
    Q_OBJECT

public:
    explicit VariantDialog(const QStringList &imagePaths, QWidget *parent = nullptr);

    // Absolute path of the chosen variant, or an empty string if cancelled.
    QString selectedPath() const;

private:
    QStringList paths;
    QListWidget *list;
    QPushButton *okButton;
};

#endif // VARIANTDIALOG_H
