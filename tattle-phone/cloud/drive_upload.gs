/**
 * Tattle Phone -> Google Drive receiver (Google Apps Script web app)
 *
 * Setup (5 minutes, free, no server to run):
 *   1. Make a Drive folder, e.g. "Tattles". Copy its ID from the URL
 *      (drive.google.com/drive/folders/<THIS PART>).
 *   2. Go to script.google.com -> New project. Paste this file in.
 *   3. Set FOLDER_ID and TOKEN below (TOKEN = same string as UPLOAD_TOKEN
 *      in the firmware's config.h).
 *   4. Deploy -> New deployment -> type "Web app".
 *        Execute as: Me          Who has access: Anyone
 *      ("Anyone" is required because the phone can't log in to Google; the
 *       TOKEN is what keeps strangers out.)
 *   5. Authorize, then copy the ".../exec" URL into UPLOAD_URL in config.h.
 *
 * If you edit this script later, use Deploy -> Manage deployments -> edit ->
 * New version, so the /exec URL stays the same.
 */

const FOLDER_ID = 'PASTE_FOLDER_ID_HERE';
const TOKEN = 'change-me-to-something-long-and-random';
const EMAIL_ME = true;          // email yourself each tattle (pings your phone)
const ATTACH_AUDIO = true;      // attach the WAV so you can play it from the email

function doPost(e) {
  try {
    if (!e || !e.parameter || e.parameter.token !== TOKEN) {
      return reply({ ok: false, error: 'bad token' });
    }
    const rawName = e.parameter.name || ('tattle_' + new Date().toISOString() + '.wav');
    const name = rawName.replace(/[^\w.\-]/g, '_');

    const bytes = Utilities.base64Decode(e.postData.contents);
    const blob = Utilities.newBlob(bytes, 'audio/wav', name);
    const file = DriveApp.getFolderById(FOLDER_ID).createFile(blob);

    if (EMAIL_ME) {
      const seconds = Math.round((bytes.length - 44) / 32000);  // 16 kHz, 16-bit mono
      MailApp.sendEmail({
        to: Session.getEffectiveUser().getEmail(),
        subject: '📞 New tattle (' + seconds + 's)',
        htmlBody: 'Someone called the tattle line!<br><a href="' + file.getUrl() +
                  '">Listen in Drive</a>',
        attachments: ATTACH_AUDIO ? [blob] : [],
      });
    }
    return reply({ ok: true, id: file.getId() });
  } catch (err) {
    return reply({ ok: false, error: String(err) });
  }
}

// Visit the /exec URL in a browser to check the deployment is alive.
function doGet() {
  return reply({ ok: true, hello: 'tattle line is open' });
}

function reply(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}
